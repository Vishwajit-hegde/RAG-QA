### pdf_parser.py - contains function for parsing a pdf into chunks of text. 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from typing import Any, List
from unstructured.cleaners.core import clean_extra_whitespace, group_broken_paragraphs
import PyPDF2
import fitz
from pymupdf_rag import to_markdown
from langchain.text_splitter import MarkdownTextSplitter

_DOCUMENT = Any
def pdf_to_text(
    file_paths: str | List[str],
    separators: List[str] = ["\n\n\n", "\n\n", "\n", " "],
    chunk_size: int = 5000,
    chunk_overlap: int = 500,
    save_as_txt: bool = True,
    txt_file_name: str = None,
    parse_func: str = 'pymupdf'
) -> List[_DOCUMENT]:
    """Loads PDF files and converts them to document chunks.

    Args:
        file_paths (str | List[str]): Path or list of paths to PDF files to be loaded.
        separators (List[str], optional): Escape sequences to treat as paragraph
            separators. Defaults to ``["\n\n\n", "\n\n"]``.
        chunk_size (int, optional): Number of tokens in each document chunk. Defaults to
            1000.
        chunk_overlap (int, optional): Number of overlapping tokens between chunks.
            Defaults to 300.

    Returns:
        List[_DOCUMENT]: List of cleaned document chunks.
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    md_splitter = MarkdownTextSplitter(chunk_size=chunk_size,
                                       chunk_overlap=chunk_overlap)
    
    if parse_func == 'unstructured':
        docs = parse_pdf_unstructured(file_paths, text_splitter)
    elif parse_func == 'pypdf':
        docs = parse_pdf_pypdf(file_paths, text_splitter)
    elif parse_func == "pymupdf":
        docs = parse_pdf_pymupdf(file_paths, md_splitter)

    if save_as_txt:
        text = [docs[i].page_content for i in range(len(docs))]
        text = "\n<CHUNK_SEPARATOR>\n".join(text)
        if txt_file_name==None:
            txt_file_name = '_'.join([file_paths[i].split('/')[-1].replace('.pdf','') for i in range(len(file_paths))]) + '.txt'
        with open("input_text_files/"+txt_file_name,'w',encoding="utf-8") as f:
            f.write(text)

    return docs

def parse_pdf_unstructured(file_paths, text_splitter):
    docs = []

    for f in file_paths:
        loader = UnstructuredFileLoader(
            f, processors=[clean_extra_whitespace, group_broken_paragraphs]
        )
        docs.extend(loader.load_and_split(text_splitter=text_splitter))

    return docs

def parse_pdf_pypdf(file_paths, text_splitter):
    docs = []

    for file_path in file_paths:
        reader = PyPDF2.PdfReader(file_path)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
            
        text = text.replace('\n',' ')
        
        docs.extend(text_splitter.create_documents([text]))

    return docs

def parse_pdf_pymupdf(file_paths,splitter):
    docs = []

    for file_path in file_paths:
        doc = fitz.open(file_path)
        md_text = to_markdown(doc)
        docs.extend(splitter.create_documents([md_text]))

    return docs

