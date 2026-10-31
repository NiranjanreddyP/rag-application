
import pymupdf
from langchain_core.documents import Document


def load_pdf(pdf_path: str) -> list[Document]:
    documents = []

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            content = page.get_text("text")

            if content.strip():
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": pdf_path,
                            "page": page_number
                        }
                    )
                )

    print(f"Total pages loaded: {len(documents)}")
    return documents

# def pdf_loader(pdf_path):

#     documents = []

#     with pymupdf.open(pdf_path) as pdf:

#         for page_number, page in enumerate(pdf, start=1):

#             content = page.get_text("text")

#             if content.strip():

#                 documents.append(
#                     Document(
#                         page_content=content,
#                         metadata={
#                             "source": pdf_path,
#                             "page": page_number
#                         }
#                     )
#                 )

#     print("Total pages loaded:", len(documents))

#     return documents