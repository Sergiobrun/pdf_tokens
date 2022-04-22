import unittest
import glob, os
import time

class Test_length(unittest.TestCase):
    def test_string(self):

        token_lenght = 10

        list_txt = []
        os.chdir("C:/Users/Usuario/PycharmProjects/biedronka/text_files/")
        for file in glob.glob("*.txt"):
            list_txt.append(file)

        for a in list_txt:
            f = open(a, 'r', encoding="utf-8")
            asd = []
            for i in f:
                asd.append(i.split())
            qwe = []
            for d in asd:
                for e in d:
                    qwe.append(e)
            print(a)
            if len(max(qwe, key=len)) > token_lenght:
                f = open(f'C:/Users/Usuario/PycharmProjects/biedronka/results.txt', "a",
                         encoding="utf-8")
                f.write(a +"   "+ max(qwe, key=len) +'\n')

            else:
                f = open(f'C:/Users/Usuario/PycharmProjects/biedronka/results.txt', "a",
                         encoding="utf-8")
                f.write(a + "   esta tranca palanca"+'\n')
            #self.assertTrue(len(max(qwe, key=len))<token_lenght, "hay una palabra con mas de 10 caracteres")