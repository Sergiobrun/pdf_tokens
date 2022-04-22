import pdfplumber
import time
import glob, os


list_pdf = []
os.chdir("C:/Users/Usuario/PycharmProjects/biedronka/pdfs/")
for file in glob.glob("*.pdf"):
    list_pdf.append(file)
print(list_pdf)


for i in list_pdf:
    timestr = time.strftime("%Y%m%d-%H%M%S")
    f = open(f'C:/Users/Usuario/PycharmProjects/biedronka/text_files/{i}{timestr}.txt', "w+", encoding="utf-8")
    with pdfplumber.open(i) as pdf:
        for pdf_page in pdf.pages:
            single_page_text = pdf_page.extract_text(x_tolerance=1)
            f.write(single_page_text)
    time.sleep(2)

