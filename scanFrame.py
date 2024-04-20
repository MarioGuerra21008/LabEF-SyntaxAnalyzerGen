#Universidad del Valle de Guatemala
#Diseño de Lenguajes de Programación - Sección: 10
#Mario Antonio Guerra Morales - Carné: 21008
#ScanFrame para generar el Scanner.py

from lexicalAnalyzerGen import *
import sys

def createScanner(regexTokens):
    i = 0
    with open("Scanner.py", "w") as file:
        file.write("from Definitions import *\n\n")
        file.write("def scan(tokenYal):\n")
        while i < len(regexTokens):
            token = regexTokens[i]
            code = regexTokens[i+1]
            file.write(f"    if tokenYal == '{token}':\n")
            if code == ' ':  # Verificar si el código consiste únicamente en espacios en blanco
                file.write("        return tokenYal\n")
            else:
                file.write("        try:\n")
                file.write(f"           {code}\n")
                file.write("        except NameError:\n")
                file.write("            print('Valor de retorno no definido.')\n")
            i += 2
        file.write(f"    return tokenYal\n")
        file.write("\n")
        file.write("def outputScanner(scannerList):\n")
        file.write("    for token, element in zip(scannerList[0], scannerList[1]):\n")
        file.write("        if token == 'undefined':\n")
        file.write("            print(f'Simbolo {element} -> Token no definido')\n")
        file.write("        else:\n")
        file.write("            scanSymbol = scan(token)\n")
        file.write("            print(f'Simbolo {element} -> Token {scanSymbol}')\n")
        file.close()
