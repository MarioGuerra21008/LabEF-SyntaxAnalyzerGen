from Definitions import *

def scan(tokenYal):
    if tokenYal == 'WS':
        return tokenYal
    if tokenYal == 'ID':
        try:
           return ID
        except NameError:
            print('Valor de retorno no definido.')
    if tokenYal == 'NUMBER':
        try:
           return NUMBER
        except NameError:
            print('Valor de retorno no definido.')
    if tokenYal == 'SEMICOLON':
        try:
           return SEMICOLON
        except NameError:
            print('Valor de retorno no definido.')
    if tokenYal == 'ASSIGNOP':
        try:
           return ASSIGNOP
        except NameError:
            print('Valor de retorno no definido.')
    if tokenYal == 'LT':
        try:
           return LT
        except NameError:
            print('Valor de retorno no definido.')
    if tokenYal == 'EQ':
        try:
           return EQ
        except NameError:
            print('Valor de retorno no definido.')
    if tokenYal == 'PLUS':
        try:
           return PLUS
        except NameError:
            print('Valor de retorno no definido.')
    if tokenYal == 'MINUS':
        try:
           return MINUS
        except NameError:
            print('Valor de retorno no definido.')
    if tokenYal == 'TIMES':
        try:
           return TIMES
        except NameError:
            print('Valor de retorno no definido.')
    if tokenYal == 'DIV':
        try:
           return DIV
        except NameError:
            print('Valor de retorno no definido.')
    if tokenYal == 'LPAREN':
        try:
           return LPAREN
        except NameError:
            print('Valor de retorno no definido.')
    if tokenYal == 'RPAREN':
        try:
           return RPAREN
        except NameError:
            print('Valor de retorno no definido.')
    return tokenYal

def outputScanner(scannerList):
    for token, element in zip(scannerList[0], scannerList[1]):
        if token == 'undefined':
            print(f'Simbolo {element} -> Token no definido')
        else:
            scanSymbol = scan(token)
            print(f'Simbolo {element} -> Token {scanSymbol}')
