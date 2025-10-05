#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR API router - provides file upload and OCR text extraction functionality
"""

import os
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from app.service.paddle_ocr_service import ocr_service
from app.service.text_utils import extract_text_from_pdf

logger = logging.getLogger(__name__)

# Create OCR router
router = APIRouter(prefix="/api/ocr", tags=["OCR"])

# Supported file types
SUPPORTED_FILE_TYPES = {
    'application/pdf': '.pdf',
    'image/jpeg': '.jpg',
    'image/jpg': '.jpg', 
    'image/png': '.png',
    'image/bmp': '.bmp',
    'image/tiff': '.tiff'
}

@router.post("/upload-and-extract")
async def upload_and_extract_text(
    file: UploadFile = File(...),
    page_range: Optional[str] = Form(None, description="Page range, e.g. '1-3' or '1,3,5', default all pages"),
    extract_tables: bool = Form(False, description="Whether to extract tables"),
    confidence_threshold: float = Form(0.5, description="OCR confidence threshold")
):
    """
    Upload file and extract text using OCR
    
    Args:
        file: Uploaded file
        page_range: Page range, supports formats: '1-3' (continuous pages) or '1,3,5' (specific pages)
        extract_tables: Whether to extract tables
        confidence_threshold: OCR confidence threshold
    
    Returns:
        JSON response containing extracted text and metadata
    """
    try:
        # Check file type
        if file.content_type not in SUPPORTED_FILE_TYPES:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}. Supported types: {list(SUPPORTED_FILE_TYPES.keys())}"
            )
        
        # Check OCR service availability
        if not ocr_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="OCR service unavailable, please check PaddleOCR configuration"
            )
        
        # Save uploaded file
        file_extension = SUPPORTED_FILE_TYPES[file.content_type]
        temp_file_path = f"temp_upload_{file.filename}{file_extension}"
        
        try:
            # Save file content
            with open(temp_file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            logger.info(f"File uploaded successfully: {file.filename}, size: {len(content)} bytes")
            
            # Process file based on type
            if file.content_type == 'application/pdf':
                result = await process_pdf_file(temp_file_path, page_range, extract_tables, confidence_threshold)
            else:
                result = await process_image_file(temp_file_path, confidence_threshold)
            
            # Add file metadata
            result.update({
                "filename": file.filename,
                "file_size": len(content),
                "file_type": file.content_type,
                "processing_time": result.get("processing_time", 0)
            })
            
            return JSONResponse(content=result)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file: {str(e)}")
                    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

async def process_pdf_file(file_path: str, page_range: Optional[str], extract_tables: bool, confidence_threshold: float) -> Dict[str, Any]:
    """Process PDF file"""
    import time
    start_time = time.time()
    
    try:
        # Parse page range
        pages_to_process = parse_page_range(page_range)
        
        # Extract text using text_utils
        pages_data = extract_text_from_pdf(file_path)
        
        # Filter by page range
        if pages_to_process:
            pages_data = [page for page in pages_data if page['page_num'] in pages_to_process]
        
        # Process results
        result = {
            "total_pages": len(pages_data),
            "pages": [],
            "full_text": "",
            "tables": [],
            "processing_time": 0
        }
        
        all_texts = []
        all_tables = []
        
        for page_data in pages_data:
            page_result = {
                "page_num": page_data['page_num'],
                "text": page_data.get('text', ''),
                "text_length": len(page_data.get('text', '')),
                "tables_count": len(page_data.get('tables', []))
            }
            
            if extract_tables:
                page_result["tables"] = page_data.get('tables', [])
                all_tables.extend(page_data.get('tables', []))
            
            result["pages"].append(page_result)
            all_texts.append(page_data.get('text', ''))
        
        # Merge all text
        result["full_text"] = '\n\n'.join(all_texts)
        result["full_text_length"] = len(result["full_text"])
        
        if extract_tables:
            result["tables"] = all_tables
            result["total_tables"] = len(all_tables)
        
        result["processing_time"] = time.time() - start_time
        
        return result
        
    except Exception as e:
        logger.error(f"PDF processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

async def process_image_file(file_path: str, confidence_threshold: float) -> Dict[str, Any]:
    """Process image file"""
    import time
    start_time = time.time()
    
    try:
        # Use PaddleOCR to process image
        ocr_result = ocr_service.recognize_text(file_path, 1)
        
        result = {
            "text": ocr_result.get('text', ''),
            "text_length": len(ocr_result.get('text', '')),
            "tables": ocr_result.get('tables', []),
            "tables_count": len(ocr_result.get('tables', [])),
            "processing_time": time.time() - start_time
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

def parse_page_range(page_range: Optional[str]) -> Optional[List[int]]:
    """Parse page range string"""
    if not page_range:
        return None
    
    try:
        pages = []
        for part in page_range.split(','):
            part = part.strip()
            if '-' in part:
                # Handle range, e.g. "1-3"
                start, end = map(int, part.split('-'))
                pages.extend(range(start, end + 1))
            else:
                # Handle single page
                pages.append(int(part))
        return sorted(set(pages))  # Remove duplicates and sort
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid page range format")

@router.get("/status")
async def get_ocr_status():
    """Get OCR service status"""
    return {
        "ocr_available": ocr_service.is_available(),
        "supported_file_types": list(SUPPORTED_FILE_TYPES.keys()),
        "max_file_size": "50MB",  # Can be adjusted as needed
        "supported_languages": ["Chinese", "English"]
    }

@router.post("/extract-from-url")
async def extract_from_url(
    url: str = Form(..., description="File URL"),
    page_range: Optional[str] = Form(None, description="Page range"),
    extract_tables: bool = Form(False, description="Whether to extract tables")
):
    """
    Download file from URL and perform OCR extraction
    
    Args:
        url: File URL
        page_range: Page range
        extract_tables: Whether to extract tables
    
    Returns:
        JSON response containing extracted text
    """
    try:
        import requests
        import tempfile
        
        # Download file
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Get file type
        content_type = response.headers.get('content-type', '')
        if content_type not in SUPPORTED_FILE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {content_type}"
            )
        
        # Create temporary file
        file_extension = SUPPORTED_FILE_TYPES[content_type]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
        
        try:
            # Process file
            if content_type == 'application/pdf':
                result = await process_pdf_file(temp_file_path, page_range, extract_tables, 0.5)
            else:
                result = await process_image_file(temp_file_path, 0.5)
            
            result["source_url"] = url
            return JSONResponse(content=result)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file: {str(e)}")
                    
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download file: {str(e)}")
    except Exception as e:
        logger.error(f"URL processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"URL processing failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ocr_service": "available" if ocr_service.is_available() else "unavailable",
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }
