from typing import Union
from pypdf import PdfReader
import pandas as pd
import io
import logging

logger = logging.getLogger(__name__)


class DocumentParser:
    """Parser for extracting text from various document formats."""
    
    SUPPORTED_TYPES = {
        "application/pdf": "pdf",
        "text/plain": "txt",
        "text/csv": "csv",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
        "application/vnd.ms-excel": "xls",
        "text/markdown": "md",
    }
    
    def parse(self, content: bytes, file_type: str) -> str:
        """
        Parse document content based on file type.
        
        Args:
            content: Raw file content as bytes
            file_type: MIME type of the file
            
        Returns:
            Extracted text content
        """
        parser_type = self.SUPPORTED_TYPES.get(file_type)
        
        if parser_type == "pdf":
            return self._parse_pdf(content)
        elif parser_type in ["txt", "md"]:
            return self._parse_text(content)
        elif parser_type == "csv":
            return self._parse_csv(content)
        elif parser_type in ["xlsx", "xls"]:
            return self._parse_excel(content)
        else:
            # Try to parse as plain text
            try:
                return self._parse_text(content)
            except Exception:
                raise ValueError(f"Unsupported file type: {file_type}")
    
    def _parse_pdf(self, content: bytes) -> str:
        """Extract text from PDF."""
        try:
            reader = PdfReader(io.BytesIO(content))
            text_parts = []
            
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            return "\n\n".join(text_parts)
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            raise ValueError(f"Failed to parse PDF: {e}")
    
    def _parse_text(self, content: bytes) -> str:
        """Extract text from plain text file."""
        try:
            # Try different encodings
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    return content.decode(encoding)
                except UnicodeDecodeError:
                    continue
            
            raise ValueError("Could not decode text file")
        except Exception as e:
            logger.error(f"Error parsing text file: {e}")
            raise ValueError(f"Failed to parse text file: {e}")
    
    def _parse_csv(self, content: bytes) -> str:
        """Extract text from CSV."""
        try:
            df = pd.read_csv(io.BytesIO(content))
            return df.to_string()
        except Exception as e:
            logger.error(f"Error parsing CSV: {e}")
            raise ValueError(f"Failed to parse CSV: {e}")
    
    def _parse_excel(self, content: bytes) -> str:
        """Extract text from Excel file."""
        try:
            df = pd.read_excel(io.BytesIO(content))
            return df.to_string()
        except Exception as e:
            logger.error(f"Error parsing Excel: {e}")
            raise ValueError(f"Failed to parse Excel file: {e}")
    
    @classmethod
    def is_supported(cls, file_type: str) -> bool:
        """Check if a file type is supported."""
        return file_type in cls.SUPPORTED_TYPES
