"""
TOKENS
LEXER
"""
# CONSTANTS

DIGITS = "0123456789"


# ERRORS
class Error:
    def __init__(self, position_start, position_end, error_name, details):
        self.error_name = error_name
        self.details = details
        self.position_start = position_start
        self.position_end = position_end

    def as_string(self):
        result = f"{self.error_name}: {self.details}"
        result += f"\nFile {self.position_start.file_name}, line {self.position_start.line_number + 1}"
        return result


class IllegalCharError(Error):
    def __init__(self, position_start, position_end, details):
        super().__init__(position_start, position_end, "Illegal Character", details)


# POSITION

class Position:
    def __init__(self, index, line_number, column, file_name, file_text):
        self.index = index
        self.line_number = line_number
        self.column = column
        self.file_name = file_name
        self.file_text = file_text

    def advance(self, current_char):
        self.index += 1
        self.column += 1

        if current_char == "\n":
            self.line_number += 1
            self.column = 0

        return self

    def copy(self):
        return Position(self.index, self.line_number, self.column, self.file_name, self.file_text)


# TOKENS
# TT means token type
TT_INT = "TT_INT"
TT_FLOAT = "TT_FLOAT"
TT_MINUS = "TT_MINUS"
TT_PLUS = "TT_PLUS"
TT_MULTIPLY = "TT_MULTIPLY"
TT_DIV = "TT_DIV"
TT_LBRACKET = "TT_LBRACKET"
TT_RBRACKET = "TT_RBRACKET"


class Token:

    def __init__(self, type_, value=None):
        self.type = type_
        self.value = value

    def __repr__(self):
        if self.value:
            return f"{self.type}:{self.value}"
        return f"{self.type}"


# LEXER

class Lexer:
    def __init__(self, text, file_name):
        self.text = text
        self.position = Position(-1, 0, -1, file_name, text)
        self.current_char = None
        self.advance()
        self.file_name = file_name

    def advance(self):
        self.position.advance(self.current_char)
        self.current_char = self.text[self.position.index] if self.position.index < len(self.text) else None

    def make_tokens(self):
        tokens = []

        while self.current_char != None:
            if self.current_char in " \t":
                self.advance()
            elif self.current_char in DIGITS:
                tokens.append(self.make_number())
            elif self.current_char == "+":
                tokens.append(Token(TT_PLUS))
                self.advance()
            elif self.current_char == "-":
                tokens.append(Token(TT_MINUS))
                self.advance()
            elif self.current_char == "*":
                tokens.append(Token(TT_MULTIPLY))
                self.advance()
            elif self.current_char == "/":
                tokens.append(Token(TT_DIV))
                self.advance()
            elif self.current_char == "(":
                tokens.append(Token(TT_LBRACKET))
                self.advance()
            elif self.current_char == ")":
                tokens.append(Token(TT_RBRACKET))
                self.advance()
            else:
                position_start = self.position.copy()
                char = self.current_char
                self.advance()
                return [], IllegalCharError(position_start, self.position, "'" + char + "'")

        return tokens, None

    def make_number(self):
        number_string = ""
        decimal_count = 0

        while self.current_char != None and self.current_char in DIGITS + ".":
            if self.current_char == ".":
                if decimal_count == 1: break
                decimal_count += 1
                number_string += "."
            else:
                number_string += self.current_char
            self.advance()
        if decimal_count == 0:
            return Token(TT_INT, int(number_string))
        else:
            return Token(TT_FLOAT, float(number_string))


# RUN

def run(file_name, text):
    lexer = Lexer(file_name, text)
    tokens, error = lexer.make_tokens()
    return tokens, error
