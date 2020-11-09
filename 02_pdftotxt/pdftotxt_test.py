from pdfminer.pdfinterp import PDFResourceManager, process_pdf
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from io import StringIO
from io import open
from urllib.request import urlopen
import docx
from docx import Document

# 다음 코드는 라이브러리에서 PDF 파일을 읽을 시 사용하는 전형적인 코드 형태이므로, 필요할 때 활용하면 됨
def read_pdf_file(pdfFile):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)

    process_pdf(rsrcmgr, device, pdfFile)
    device.close()

    content = retstr.getvalue()
    retstr.close()
    return content

# pdf_file = urlopen("http://pythonscraping.com/pages/warandpeace/chapter1.pdf")  # 웹에 있는 pdf 파일을 읽을 수 있음
pdf_file = open("byte_of_python.pdf", "rb")         # 로컬 PC에 있는 pdf 파일도 읽을 수 있음
contents = read_pdf_file(pdf_file)
#print(contents)

contents.save('test.docx')

pdf_file.close()