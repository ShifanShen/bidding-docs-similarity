"""
下载Bootstrap资源到本地
用于解决CDN不稳定问题
"""
import os
import sys
from pathlib import Path

# 尝试使用requests库（如果可用），否则使用urllib
try:
    import requests
    USE_REQUESTS = True
except ImportError:
    import urllib.request
    import urllib.error
    USE_REQUESTS = False

# Bootstrap版本
BOOTSTRAP_VERSION = "5.3.2"

# 资源URL和本地路径映射
RESOURCES = {
    "css": {
        "url": f"https://cdn.jsdelivr.net/npm/bootstrap@{BOOTSTRAP_VERSION}/dist/css/bootstrap.min.css",
        "local": f"app/static/libs/bootstrap/{BOOTSTRAP_VERSION}/bootstrap.min.css"
    },
    "js": {
        "url": f"https://cdn.jsdelivr.net/npm/bootstrap@{BOOTSTRAP_VERSION}/dist/js/bootstrap.bundle.min.js",
        "local": f"app/static/libs/bootstrap/{BOOTSTRAP_VERSION}/bootstrap.bundle.min.js"
    }
}

# 备用CDN列表
FALLBACK_CDNS = [
    "https://cdn.jsdelivr.net",
    "https://cdn.bootcdn.net",
    "https://unpkg.com",
    "https://cdnjs.cloudflare.com"
]


def get_cdn_url(base_url, resource_type):
    """根据资源类型和CDN基础URL生成完整URL"""
    if "jsdelivr" in base_url:
        return f"https://cdn.jsdelivr.net/npm/bootstrap@{BOOTSTRAP_VERSION}/dist/{resource_type}/bootstrap.{'bundle.' if resource_type == 'js' else ''}min.{resource_type}"
    elif "bootcdn" in base_url:
        return f"https://cdn.bootcdn.net/ajax/libs/bootstrap/{BOOTSTRAP_VERSION}/{resource_type}/bootstrap.{'bundle.' if resource_type == 'js' else ''}min.{resource_type}"
    elif "unpkg" in base_url:
        return f"https://unpkg.com/bootstrap@{BOOTSTRAP_VERSION}/dist/{resource_type}/bootstrap.{'bundle.' if resource_type == 'js' else ''}min.{resource_type}"
    elif "cdnjs" in base_url:
        return f"https://cdnjs.cloudflare.com/ajax/libs/bootstrap/{BOOTSTRAP_VERSION}/{resource_type}/bootstrap.{'bundle.' if resource_type == 'js' else ''}min.{resource_type}"
    return base_url


def download_file(url, local_path):
    """下载文件"""
    try:
        print(f"正在下载: {url}")
        print(f"保存到: {local_path}")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # 使用requests或urllib下载
        if USE_REQUESTS:
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            urllib.request.urlretrieve(url, local_path)
        
        # 检查文件大小
        file_size = os.path.getsize(local_path)
        if file_size < 1000:  # 如果文件太小，可能是错误页面
            os.remove(local_path)  # 删除无效文件
            raise Exception(f"下载的文件可能无效（大小: {file_size} 字节）")
        
        print(f"✓ 下载成功 ({file_size:,} 字节)")
        return True
    except Exception as e:
        error_type = type(e).__name__
        print(f"✗ 下载失败 [{error_type}]: {e}")
        # 如果文件存在但可能无效，尝试删除
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
            except:
                pass
        return False


def download_with_fallback(resource_type):
    """使用备用CDN下载资源"""
    # 首先尝试主CDN
    main_url = RESOURCES[resource_type]["url"]
    local_path = RESOURCES[resource_type]["local"]
    
    if download_file(main_url, local_path):
        return True
    
    # 尝试备用CDN
    print(f"\n主CDN失败，尝试备用CDN...")
    for cdn_base in FALLBACK_CDNS[1:]:  # 跳过第一个（已经试过了）
        fallback_url = get_cdn_url(cdn_base, resource_type)
        if download_file(fallback_url, local_path):
            return True
    
    return False


def main():
    """主函数"""
    print("=" * 60)
    print("Bootstrap 资源本地化工具")
    print("=" * 60)
    print(f"版本: {BOOTSTRAP_VERSION}\n")
    
    success_count = 0
    
    # 下载CSS
    print("\n[1/2] 下载 Bootstrap CSS...")
    if download_with_fallback("css"):
        success_count += 1
    else:
        print("⚠ CSS下载失败，将使用CDN备用方案")
    
    # 下载JS
    print("\n[2/2] 下载 Bootstrap JS...")
    if download_with_fallback("js"):
        success_count += 1
    else:
        print("⚠ JS下载失败，将使用CDN备用方案")
    
    # 总结
    print("\n" + "=" * 60)
    if success_count == 2:
        print("✓ 所有资源下载成功！")
        print(f"本地资源路径: app/static/libs/bootstrap/{BOOTSTRAP_VERSION}/")
        print("\n现在可以修改HTML文件使用本地资源了。")
    elif success_count == 1:
        print("⚠ 部分资源下载失败，请检查网络连接后重试。")
    else:
        print("✗ 所有资源下载失败，请检查网络连接。")
        print("页面将自动回退到CDN备用方案。")
    print("=" * 60)


if __name__ == "__main__":
    main()
