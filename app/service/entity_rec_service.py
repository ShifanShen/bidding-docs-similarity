
from __future__ import annotations

import copy
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import hanlp

from app.config.entity_rec_config import (
    DEFAULT_ENTITIES,
    ESC_TOKENS,
    SIGN_WORDS,
    COMPANY_SUFFIX_STRONG,
    GENERIC_ORG_EXACT,
    GENERIC_ORG_PATTERN,
    RE_GOV_DEPT,
    RE_NAME_KV,
    RE_COMPANY_STRONG,
    RE_PHONE,
    RE_IDCARD,
)

# =========================
# 1) 工具：文本清理/切分
# =========================

def strip_literal_esc_tokens(s: str) -> str:
    """只删除字面量 \n（两个字符：反斜杠 + n）"""
    if not s:
        return s
    # 只按配置删（当前就是 ("\\n",)）
    for t in ESC_TOKENS:
        s = s.replace(t, "")
    return s


def normalize_ws(s: str) -> str:
    """压缩真实空白（不会影响字面量\n，因为已在外层删掉）"""
    return re.sub(r"\s+", " ", s).strip()


def iter_text_nodes(payload: Any) -> Iterable[Dict[str, Any]]:
    """遍历 payload 中所有包含 'text' 字段的 dict 节点"""
    if isinstance(payload, dict):
        if isinstance(payload.get("text"), str):
            yield payload
        for v in payload.values():
            yield from iter_text_nodes(v)
    elif isinstance(payload, list):
        for it in payload:
            yield from iter_text_nodes(it)


def chunk_text(text: str, max_chars: int = 1800) -> List[str]:
    """长文本切块，避免 HanLP OOM；尽量在标点处切"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    seps = set("。！？；;,，、\n")
    out: List[str] = []
    i = 0
    n = len(text)

    while i < n:
        j = min(i + max_chars, n)
        cut = -1
        for k in range(j, max(i + int(max_chars * 0.6), i + 1), -1):
            if text[k - 1] in seps:
                cut = k
                break
        if cut == -1:
            cut = j
        out.append(text[i:cut].strip())
        i = cut

    return [x for x in out if x]


# =========================
# 2) HanLP：模型加载 + NER解析
# =========================

def load_hanlp_mtl_model():
    """
    默认 CPU（更稳）：ENTITY_REC_USE_GPU=1 才用 GPU
    """
    use_gpu = os.getenv("ENTITY_REC_USE_GPU", "0") == "1"
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    try:
        nlp = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
    except Exception:
        nlp = hanlp.load(hanlp.pretrained.mtl.OPEN_TOK_NER_ELECTRA_SMALL_ZH)

    try:
        nlp.eval()
    except Exception:
        pass

    return nlp


def pick_ner_key(doc: Dict[str, Any]) -> Optional[str]:
    for k in ("ner/msra", "ner/pku", "ner", "ner/ontonotes"):
        if k in doc:
            return k
    for k in doc.keys():
        if k.startswith("ner"):
            return k
    return None


def find_tok_key(doc: Dict[str, Any]) -> Optional[str]:
    for k in ("tok/fine", "tok/coarse", "tok"):
        if k in doc:
            return k
    for k in doc.keys():
        if k.startswith("tok"):
            return k
    return None


def ner_spans_from_doc(doc: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    统一输出：(entity_text, label)
    兼容 span list 与 BIO tags 两类输出
    """
    ner_key = pick_ner_key(doc)
    if not ner_key:
        return []

    ner_out = doc[ner_key]

    # Case A: span list
    if isinstance(ner_out, list) and ner_out and isinstance(ner_out[0], (tuple, list)) and len(ner_out[0]) >= 4:
        return [(str(it[0]).strip(), str(it[1]).strip()) for it in ner_out]

    # Case B: BIO tags
    tok_key = find_tok_key(doc)
    if not tok_key:
        return []
    tok = doc[tok_key]
    if not (isinstance(tok, list) and isinstance(ner_out, list) and len(tok) == len(ner_out)):
        return []

    def bio_to_spans(tags: List[str]) -> List[Tuple[int, int, str]]:
        spans = []
        start = None
        cur = None

        def close(i: int):
            nonlocal start, cur
            if start is not None and cur is not None:
                spans.append((start, i, cur))
            start, cur = None, None

        for i, tg in enumerate(tags):
            if not tg or tg == "O":
                close(i)
                continue
            if tg.startswith("B-"):
                close(i)
                cur = tg[2:]
                start = i
                continue
            if tg.startswith("I-"):
                lab = tg[2:]
                if cur is None:
                    cur = lab
                    start = i
                elif lab != cur:
                    close(i)
                    cur = lab
                    start = i
                continue
            close(i)
            cur = tg
            start = i

        close(len(tags))
        return spans

    out: List[Tuple[str, str]] = []
    for s, e, lab in bio_to_spans(ner_out):
        out.append(("".join(tok[s:e]).strip(), lab.strip()))
    return out


def normalize_label(label: str) -> str:
    lab = (label or "").upper()
    if "PER" in lab or lab == "PERSON" or lab == "NR":
        return "PERSON"
    if "ORG" in lab or lab == "ORGANIZATION" or lab == "NT":
        return "ORG"
    return lab


# =========================
# 3) 清洗/过滤：人名、公司（只要“投标公司”）
# =========================

INVALID_NAME_PHRASE = re.compile(r"(已接收|已收到|已签收|已签|收悉|确认|同意|盖章|签章|签名|签字|日期|电话|身份证|地址|单位|公司)")

def clean_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"[（(]\s*(?:签名|盖章|签字)\s*[）)]", "", name).strip()
    name = re.sub(r"[_＿—\-﹍﹎﹏\s]+", "", name).strip()

    if not re.fullmatch(r"[\u4e00-\u9fff]{2,10}(?:·[\u4e00-\u9fff]{1,10})?", name):
        return ""
    if INVALID_NAME_PHRASE.search(name):
        return ""
    return name


def is_generic_org_name(name: str) -> bool:
    if not name:
        return True
    if name in GENERIC_ORG_EXACT:
        return True
    if GENERIC_ORG_PATTERN.match(name):
        return True
    return False


def clean_company(comp: str) -> str:
    comp = (comp or "").strip()
    comp = comp.strip("：:;；，,。. \t")
    comp = strip_literal_esc_tokens(comp)
    comp = re.sub(r"\s+", "", comp)

    comp = re.sub(r"[（(]\s*(?:公章|盖章|签章|签名|签字|法人章)\s*[）)]", "", comp).strip()
    comp = comp.strip("：:;；，,。. ")

    if len(comp) < 4:
        return ""
    if is_generic_org_name(comp):
        return ""

    # 去签章词
    if any(w and (w in comp) for w in SIGN_WORDS):
        return ""
    return comp


def is_bid_company_name(name: str) -> bool:
    """
    “公司实体”只保留招投标公司名称：
    - 命中强公司后缀：必收（有限公司/股份有限公司/...）
    - 或者以“公司”结尾：可收（弱后缀）
    - 但若以政府部门/事业单位后缀结尾：排除
    """
    if not name:
        return False

    # 强公司后缀：直接认为是公司
    if any(suf in name for suf in COMPANY_SUFFIX_STRONG) or RE_COMPANY_STRONG.search(name):
        return True

    # 弱公司后缀：以“公司”结尾
    if name.endswith("公司"):
        # 末尾是政府/部门/事业单位/中心等：排除
        if RE_GOV_DEPT.search(name):
            return False
        return True

    # 其他（管理局/委员会/中心/医院/学校等）不算投标公司
    return False


def dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# =========================
# 4) 核心：从单段 text 抽实体（分块 + HanLP + 正则）
# =========================

class EntityRecService:
    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = load_hanlp_mtl_model()
        return self._nlp

    def extract_entities_from_text(
            self,
            raw_text: str,
            entity_keys: Optional[List[str]] = None,
            keep_offsets: bool = False,
            max_chars_per_chunk: int = 1800,
    ) -> List[Dict[str, Any]]:
        """
        输出：entities 数组
        """
        entity_keys = entity_keys or DEFAULT_ENTITIES
        entities: List[Dict[str, Any]] = []

        if not raw_text or not raw_text.strip():
            return entities

        # 只删字面量 \n；再压缩真实空白
        view_text = normalize_ws(strip_literal_esc_tokens(raw_text))

        # ========== 1) HanLP（人名 / 公司）分块推理 ==========
        persons: List[str] = []
        companies: List[str] = []

        for ch in chunk_text(view_text, max_chars=max_chars_per_chunk):
            doc = self.nlp(ch)
            if not isinstance(doc, dict):
                continue

            for ent_text, lab in ner_spans_from_doc(doc):
                lab2 = normalize_label(lab)

                if lab2 == "PERSON" and "人名" in entity_keys:
                    nm = clean_name(ent_text)
                    if nm:
                        persons.append(nm)

                elif lab2 == "ORG" and "公司" in entity_keys:
                    cp = clean_company(ent_text)
                    if cp and is_bid_company_name(cp):
                        companies.append(cp)

            # KV兜底：提示词 + 冒号后的名字
            if "人名" in entity_keys:
                for m in RE_NAME_KV.finditer(ch):
                    nm = clean_name(m.group("value"))
                    if nm:
                        persons.append(nm)

            # 公司兜底：仅强公司后缀（不再把管理局/委员会/中心之类塞进“公司”）
            if "公司" in entity_keys:
                for m in RE_COMPANY_STRONG.finditer(ch):
                    cp = clean_company(m.group(0))
                    if cp and is_bid_company_name(cp):
                        companies.append(cp)

        persons = dedup_keep_order(persons)
        companies = dedup_keep_order(companies)

        # ========== 2) 正则（联系方式 / 身份证） ==========
        phones: List[str] = []
        idcards: List[str] = []

        if "联系方式" in entity_keys:
            for m in RE_PHONE.finditer(view_text):
                # 保留原座机格式：仅去真实空白
                ph = re.sub(r"\s+", "", m.group(0)).strip()
                if ph:
                    phones.append(ph)
            phones = dedup_keep_order(phones)

        if "身份证" in entity_keys:
            # 允许被空白分隔：先去空白再匹配
            compact = re.sub(r"\s+", "", view_text)
            for m in RE_IDCARD.finditer(compact):
                idcards.append(m.group(0).upper())
            idcards = dedup_keep_order(idcards)

        # ========== 3) 组装输出 ==========
        def add_items(label_cn: str, values: List[str]):
            for v in values:
                item = {"entity": label_cn, "text_content": v}
                if keep_offsets:
                    pos = view_text.find(v)
                    item["start"] = pos if pos >= 0 else -1
                    item["end"] = (pos + len(v)) if pos >= 0 else -1
                entities.append(item)

        if "公司" in entity_keys:
            add_items("公司", companies)
        if "人名" in entity_keys:
            add_items("人名", persons)
        if "联系方式" in entity_keys:
            add_items("联系方式", phones)
        if "身份证" in entity_keys:
            add_items("身份证", idcards)

        return entities

    def enrich_extracted_data(
            self,
            payload: Any,
            entity_keys: Optional[List[str]] = None,
            keep_offsets: bool = False,
            override_existing: bool = True,
            copy_input: bool = True,
    ) -> Any:
        """
        输入：extracted_data JSON（或 JSON 字符串）
        输出：同结构 JSON，在每个 text 节点补 entities: [...]
        """
        if isinstance(payload, str):
            payload_obj = json.loads(payload)
        else:
            payload_obj = payload

        if not isinstance(payload_obj, (dict, list)):
            raise ValueError("payload 必须是 JSON 对象或 JSON 数组")

        data = copy.deepcopy(payload_obj) if copy_input else payload_obj

        for node in iter_text_nodes(data):
            txt = node.get("text", "")
            if not isinstance(txt, str):
                continue

            if (not override_existing) and isinstance(node.get("entities"), list) and node["entities"]:
                continue

            node["entities"] = self.extract_entities_from_text(
                raw_text=txt,
                entity_keys=entity_keys,
                keep_offsets=keep_offsets,
            )

        return data

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        兼容接口：从文本中提取实体
        """
        return self.extract_entities_from_text(raw_text=text)

    def recognize(self, text: str) -> List[Dict[str, Any]]:
        """
        兼容接口：识别文本中的实体
        """
        return self.extract_entities_from_text(raw_text=text)

    def extract_entities_from_json(self, json_data: Dict) -> List[Dict]:
        """
        从相似度分析的JSON结果中提取实体
        保持原始text字段不变
        """
        results = []

        if isinstance(json_data, dict):
            # 单个文档
            if "text" in json_data:
                entities = self.extract_entities(json_data["text"])
                result = {
                    "page": json_data.get("page"),
                    "text": json_data["text"],  # 保持原始text不变
                    "entities": entities
                }
                results.append(result)

        elif isinstance(json_data, list):
            # 多个文档
            for doc in json_data:
                if isinstance(doc, dict) and "text" in doc:
                    entities = self.extract_entities(doc["text"])
                    result = {
                        "page": doc.get("page"),
                        "text": doc["text"],  # 保持原始text不变
                        "entities": entities
                    }
                    results.append(result)

        return results


def extract_entities_from_text(raw_text: str) -> List[Dict[str, Any]]:
    """外部调用的函数，直接使用默认服务实例"""
    return default_entity_rec_service.extract_entities_from_text(raw_text)


def extract_entities_from_json(json_data: Dict) -> List[Dict]:
    """外部调用的函数，直接使用默认服务实例"""
    return default_entity_rec_service.extract_entities_from_json(json_data)


def extract_entities(text: str) -> List[Dict[str, Any]]:
    """外部调用的函数，直接使用默认服务实例"""
    return default_entity_rec_service.extract_entities(text)


def recognize(text: str) -> List[Dict[str, Any]]:
    """外部调用的函数，直接使用默认服务实例"""
    return default_entity_rec_service.recognize(text)


def enrich_extracted_data(payload: Any) -> Any:
    """外部调用的函数，直接使用默认服务实例"""
    return default_entity_rec_service.enrich_extracted_data(payload)


# 全局实例
default_entity_rec_service = EntityRecService()
