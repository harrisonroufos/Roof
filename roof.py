"""
TOKENS
LEXER
"""
from string_with_arrows import *

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
        result += "\n\n" + string_with_arrows(self.position_start.file_text, self.position_start, self.position_end)

        return result


class IllegalCharError(Error):
    def __init__(self, position_start, position_end, details):
        super().__init__(position_start, position_end, "Illegal Character", details)


class InvalidSyntaxError(Error):
    def __init__(self, position_start, position_end, details):
        super().__init__(position_start, position_end, "InvalidSyntaxError", details)


# POSITION

class Position:
    def __init__(self, index, line_number, column, file_name, file_text):
        self.index = index
        self.line_number = line_number
        self.column = column
        self.file_name = file_name
        self.file_text = file_text

    def advance(self, current_char=None):
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
TT_INT = "INT"
TT_FLOAT = "FLOAT"
TT_MINUS = "MINUS"
TT_PLUS = "PLUS"
TT_MULTIPLY = "MULTIPLY"
TT_DIV = "DIV"
TT_LBRACKET = "LBRACKET"
TT_RBRACKET = "RBRACKET"
TT_END_OF_FILE = "END_OF_FILE"


class Token:
    def __init__(self, type_, value=None, position_start=None, position_end=None):
        self.type = type_
        self.value = value

        if position_start:
            self.position_start = position_start.copy()
            self.position_end = position_start.copy()
            self.position_end.advance()

        if position_end:
            self.position_end = position_end

    def __repr__(self):
        if self.value:
            return f'{self.type}:{self.value}'
        return f'{self.type}'


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
                tokens.append(Token(TT_PLUS, position_start=self.position))
                self.advance()
            elif self.current_char == "-":
                tokens.append(Token(TT_MINUS, position_start=self.position))
                self.advance()
            elif self.current_char == "*":
                tokens.append(Token(TT_MULTIPLY, position_start=self.position))
                self.advance()
            elif self.current_char == "/":
                tokens.append(Token(TT_DIV, position_start=self.position))
                self.advance()
            elif self.current_char == "(":
                tokens.append(Token(TT_LBRACKET, position_start=self.position))
                self.advance()
            elif self.current_char == ")":
                tokens.append(Token(TT_RBRACKET, position_start=self.position))
                self.advance()
            else:
                position_start = self.position.copy()
                char = self.current_char
                self.advance()
                return [], IllegalCharError(position_start, self.position, "'" + char + "'")
        tokens.append(Token(TT_END_OF_FILE))
        return tokens, None

    def make_number(self):
        number_string = ""
        decimal_count = 0
        position_start = self.position.copy()

        while self.current_char != None and self.current_char in DIGITS + ".":
            if self.current_char == ".":
                if decimal_count == 1:
                    break
                decimal_count += 1
                number_string += "."
            else:
                number_string += self.current_char
            self.advance()
        if decimal_count == 0:
            return Token(TT_INT, int(number_string), position_start, self.position)
        else:
            return Token(TT_FLOAT, float(number_string), position_start, self.position)


# NODES

class NumberNode:
    def __init__(self, token):
        self.token = token

    def __repr__(self):
        return f"{self.token}"


class BinaryOperationNode:

    def __init__(self, left_node, operator_token, right_node):
        self.left_node = left_node
        self.operator_token = operator_token
        self.right_node = right_node

    def __repr__(self):
        return f"({self.left_node}, {self.operator_token}, {self.right_node})"


# PARSER RESULTS

class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None

    def register(self, result):
        if isinstance(result, ParseResult):
            if result.error:
                self.error = result.error
            return result.node

        return result

    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        self.error = error
        return self


# PARSER
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.token_index = -1
        self.advance()

    def advance(self):
        self.token_index += 1
        if self.token_index < len(self.tokens):
            self.current_token = self.tokens[self.token_index]
            return self.current_token

    def parse(self):
        result = self.expression()
        if not result.error and self.current_token.type != TT_END_OF_FILE:
            return result.failure(InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end,
                                                     "Expected '+', '-', '*' or '/'"))
        return result

    def factor(self):
        result = ParseResult()
        token = self.current_token

        if token.type in (TT_INT, TT_FLOAT):
            result.register(self.advance())
            return result.success(NumberNode(token))
        return result.failure(InvalidSyntaxError(token.position_start, token.position_end, "Expected int or float"))

    def term(self):
        return self.binary_operation(self.factor, (TT_MULTIPLY, TT_DIV))

    def expression(self):
        return self.binary_operation(self.term, (TT_PLUS, TT_MINUS))

    def binary_operation(self, function, operation_tokens):
        result = ParseResult()
        left = result.register(function())
        if result.error:
            return result
        while self.current_token.type in operation_tokens:
            operator_token = self.current_token
            result.register(self.advance())
            right = result.register(function())
            if result.error:
                return result
            left = BinaryOperationNode(left, operator_token, right)

        return result.success(left)


# RUN

def run(file_name, text):
    # GENERATE TOKENS
    lexer = Lexer(file_name, text)
    tokens, error = lexer.make_tokens()
    if error:
        return None, error
    # GENERATE ABSTRACT SYNTAX TREE
    parser = Parser(tokens)
    abstract_syntax_tree = parser.parse()

    return abstract_syntax_tree.node, abstract_syntax_tree.error
