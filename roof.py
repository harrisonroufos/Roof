"""
TOKENS
LEXER
"""
import os

from string_with_arrows import *
import string

# CONSTANTS

DIGITS = "0123456789"
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS


# ERRORS
class Error:
    def __init__(self, position_start, position_end, error_name, details):
        self.position_start = position_start
        self.position_end = position_end
        self.error_name = error_name
        self.details = details

    def as_string(self):
        result = f'{self.error_name}: {self.details}\n'
        result += f'File {self.position_start.file_name}, line {self.position_start.line_number + 1}'
        result += '\n\n' + string_with_arrows(self.position_start.file_text, self.position_start, self.position_end)
        return result


class IllegalCharError(Error):
    def __init__(self, position_start, position_end, details):
        super().__init__(position_start, position_end, 'Illegal Character', details)


class InvalidSyntaxError(Error):
    def __init__(self, position_start, position_end, details):
        super().__init__(position_start, position_end, "Invalid Syntax", details)


class RTError(Error):
    def __init__(self, position_start, position_end, details, context):
        super().__init__(position_start, position_end, "Runtime Error", details)
        self.context = context

    def as_string(self):
        result = self.generate_traceback()
        result += f"{self.error_name}: {self.details}"
        result += "\n\n" + string_with_arrows(self.position_start.file_text, self.position_start, self.position_end)
        return result

    def generate_traceback(self):
        result = ""
        position = self.position_start
        context = self.context

        while context:
            result = f"File {position.file_name}, line {str(position.line_number + 1)}, in {context.display_name}\n" + result
            position = context.parent_entry_position
            context = context.parent

        return "Traceback (most recent call last):\n" + result


class ExpectedCharError:
    def __init__(self, position_start, position_end, details):
        super().__init__(position_start, position_end, "Expected Character", details)


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
TT_STRING = "STRING"
TT_MINUS = "MINUS"
TT_PLUS = "PLUS"
TT_MULTIPLY = "MULTIPLY"
TT_DIV = "DIV"
TT_LBRACKET = "LBRACKET"
TT_RBRACKET = "RBRACKET"
TT_LSQUARE = "LSQUARE"
TT_RSQUARE = "RSQUARE"
TT_END_OF_FILE = "END_OF_FILE"
TT_POWER_OF = "POWER_OF"
TT_IDENTIFIER = "IDENTIFIER"
TT_KEYWORD = "KEYWORD"
TT_EQUALS = "EQUALS"
TT_EQUALS_EQUALS = "EQUALS_EQUALS"
TT_NOT_EQUALS = "NOT_EQUALS"
TT_LESS_THAN = "LESS_THAN"
TT_GREATER_THAN = "GREATER_THAN"
TT_LESS_THAN_EQUAL = "LESS_THAN_EQUAL"
TT_GREATER_THAN_EQUAL = "GREATER_THAN_EQUAL"
TT_COMMA = "COMMA"
TT_ARROW = "ARROW"
TT_NEWLINE = "NEWLINE"

KEYWORDS = ["VAR", "AND", "OR", "NOT", "IF", "THEN", "ELIF", "ELSE", "FOR", "TO", "STEP", "WHILE", "FUNC", "END",
            "RETURN", "CONTINUE", "BREAK"]


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

    def matches(self, type_, value):
        return self.type == type_ and self.value == value

    def __repr__(self):
        if self.value: return f'{self.type}:{self.value}'
        return f'{self.type}'


# LEXER

class Lexer:
    def __init__(self, file_name, text):
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
            elif self.current_char == "#":
                self.skip_comment()
            elif self.current_char in ';\n':
                tokens.append(Token(TT_NEWLINE, position_start=self.position))
                self.advance()
            elif self.current_char in DIGITS:
                tokens.append(self.make_number())
            elif self.current_char in LETTERS:
                tokens.append(self.make_identifier())
            elif self.current_char == '"':
                tokens.append(self.make_string())
            elif self.current_char == "+":
                tokens.append(Token(TT_PLUS, position_start=self.position))
                self.advance()
            elif self.current_char == "-":
                tokens.append(self.make_minus_or_arrow())
                # tokens.append(Token(TT_MINUS, position_start=self.position))
                # self.advance()
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
            elif self.current_char == "^":
                tokens.append((Token(TT_POWER_OF, position_start=self.position)))
                self.advance()
            elif self.current_char == "[":
                tokens.append(Token(TT_LSQUARE, position_start=self.position))
                self.advance()
            elif self.current_char == "]":
                tokens.append(Token(TT_RSQUARE, position_start=self.position))
                self.advance()
            # elif self.current_char == "=":
            #     tokens.append((Token(TT_EQUALS, position_start=self.position)))
            #     self.advance()
            elif self.current_char == ",":
                tokens.append(Token(TT_COMMA, position_start=self.position))
                self.advance()
            elif self.current_char == "=":
                tokens.append(self.make_equals())
            elif self.current_char == "<":
                tokens.append(self.make_less_than())
            elif self.current_char == '>':
                tokens.append(self.make_greater_than())
            elif self.current_char == "!":
                token, error = self.make_not_equals()

                if error:
                    return [], error
                tokens.append(token)
            else:
                position_start = self.position.copy()
                char = self.current_char
                self.advance()
                return [], IllegalCharError(position_start, self.position, "'" + char + "'")
        tokens.append(Token(TT_END_OF_FILE, position_start=self.position))
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

    def make_string(self):
        string = ""
        position_start = self.position.copy()
        escape_character = False
        self.advance()

        escape_characters = {'n': '\n', 't': '\t'}

        while self.current_char != None and (self.current_char != '"' or escape_character):
            if escape_character:
                string += escape_characters.get(self.current_char, self.current_char)
            else:
                if self.current_char == "\\":
                    escape_character = True
                else:
                    string += self.current_char
            self.advance()
            escape_character = False

        self.advance()
        return Token(TT_STRING, string, position_start, self.position)

    def make_identifier(self):
        id_str = ""
        position_start = self.position.copy()

        while self.current_char != None and self.current_char in LETTERS_DIGITS + "_":
            id_str += self.current_char
            self.advance()

        token_type = TT_KEYWORD if id_str in KEYWORDS else TT_IDENTIFIER
        return Token(token_type, id_str, position_start, self.position)

    def make_not_equals(self):
        position_start = self.position.copy()
        self.advance()

        if self.current_char == "=":
            self.advance()
            return Token(TT_NOT_EQUALS, position_start=position_start, position_end=self.position), None

        self.advance()
        return None, ExpectedCharError(position_start, self.position, "'=' (after '!')")

    def make_equals(self):
        token_type = TT_EQUALS
        position_start = self.position.copy()
        self.advance()

        if self.current_char == "=":
            self.advance()
            token_type = TT_EQUALS_EQUALS

        return Token(token_type, position_start=position_start, position_end=self.position)

    def make_less_than(self):
        token_type = TT_LESS_THAN
        position_start = self.position.copy()
        self.advance()

        if self.current_char == "=":
            self.advance()
            token_type = TT_LESS_THAN_EQUAL

        return Token(token_type, position_start=position_start, position_end=self.position)

    def make_greater_than(self):
        token_type = TT_GREATER_THAN
        position_start = self.position.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            tok_type = TT_GREATER_THAN_EQUAL

        return Token(token_type, position_start=position_start, position_end=self.position)

    def make_minus_or_arrow(self):
        token_type = TT_MINUS
        position_start = self.position.copy()
        self.advance()

        if self.current_char == ">":
            self.advance()
            token_type = TT_ARROW

        return Token(token_type, position_start=position_start, position_end=self.position)

    def skip_comment(self):
        self.advance()
        while self.current_char != "\n":
            self.advance()
        self.advance()


# NODES

class NumberNode:
    def __init__(self, token):
        self.token = token

        self.position_start = self.token.position_start
        self.position_end = self.token.position_end

    def __repr__(self):
        return f"{self.token}"


class ListNode:
    def __init__(self, element_nodes, position_start, position_end):
        self.element_node = element_nodes

        self.position_start = position_start
        self.position_end = position_end


class StringNode:
    def __init__(self, token):
        self.token = token

        self.position_start = self.token.position_start
        self.position_end = self.token.position_end

    def __repr__(self):
        return f"{self.token}"


class VariableAccessNode:
    def __init__(self, var_name_token):
        self.var_name_token = var_name_token

        self.position_start = self.var_name_token.position_start
        self.position_end = self.var_name_token.position_end


class VariableAssignNode:
    def __init__(self, var_name_token, value_node):
        self.var_name_token = var_name_token
        self.value_node = value_node

        self.position_start = self.var_name_token.position_start
        self.position_end = self.value_node.position_end


class BinaryOperationNode:

    def __init__(self, left_node, operator_token, right_node):
        self.left_node = left_node
        self.operator_token = operator_token
        self.right_node = right_node

        self.position_start = self.left_node.position_start
        self.position_end = self.right_node.position_end

    def __repr__(self):
        return f"({self.left_node}, {self.operator_token}, {self.right_node})"


class UnaryOperationNode:

    def __init__(self, operator_token, node):
        self.operator_token = operator_token
        self.node = node

        self.position_start = self.operator_token.position_start
        self.position_end = self.operator_token.position_end

    def __repr__(self):
        return f"{self.operator_token}, {self.node}"


class IfNode:
    def __init__(self, cases, else_case):
        self.cases = cases
        self.else_case = else_case

        self.position_start = self.cases[0][0]
        self.position_end = (self.else_case or self.cases[len(self.cases) - 1])[0].position_end


class ForNode:
    def __init__(self, var_name_token, start_value_node, end_value_node, step_value_node, body_node,
                 should_return_null):
        self.var_name_token = var_name_token
        self.start_value_node = start_value_node
        self.end_value_node = end_value_node
        self.step_value_node = step_value_node
        self.body_node = body_node
        self.should_return_null = should_return_null

        self.position_start = self.var_name_token.position_start
        self.position_end = self.body_node.position_end


class WhileNode:
    def __init__(self, condition_node, body_node, should_return_null):
        self.condition_node = condition_node
        self.body_node = body_node
        self.should_return_null = should_return_null

        self.position_start = self.condition_node.position_start
        self.position_end = self.body_node.position_end


class FuncDefNode:
    def __init__(self, var_name_token, arg_name_tokens, body_node, should_auto_return):
        self.var_name_token = var_name_token
        self.arg_name_tokens = arg_name_tokens
        self.body_node = body_node
        self.should_auto_return = should_auto_return

        if self.var_name_token:
            self.position_start = self.var_name_token.position_start
        elif len(self.arg_name_tokens) > 0:
            self.position_start = self.arg_name_tokens[0].position_start
        else:
            self.position_start = self.body_node.position_start

        self.position_end = self.body_node.position_end


class CallNode:
    def __init__(self, node_to_call, arg_nodes):
        self.node_to_call = node_to_call
        self.arg_nodes = arg_nodes

        self.position_start = self.node_to_call.position_start

        if len(self.arg_nodes) > 0:
            self.position_end = self.arg_nodes[len(self.arg_nodes) - 1].position_end
        else:
            self.position_end = self.node_to_call.position_end


class ReturnNode:
    def __init__(self, node_to_return, position_start, position_end):
        self.node_to_return = node_to_return
        self.position_start = position_start
        self.position_end = position_end


class ContinueNode:
    def __init__(self, position_start, position_end):
        self.position_start = position_start
        self.position_end = position_end


class BreakNode:
    def __init__(self, position_start, position_end):
        self.position_start = position_start
        self.position_end = position_end


# PARSER RESULTS

class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
        self.advance_count = 0
        self.to_reverse_count = 0

    def register_advancement(self):
        self.advance_count += 1

    def register(self, result):
        self.advance_count += result.advance_count
        if result.error:
            self.error = result.error
        return result.node

    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        if not self.error or self.advance_count == 0:
            self.error = error
        return self

    def try_register(self, result):
        if result.error:
            self.to_reverse_count = result.advance_count
            return None
        return self.register(result)


# PARSER
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.token_index = -1
        self.advance()

    def advance(self):
        self.token_index += 1
        self.update_current_token()
        return self.current_token

    def reverse(self, amount=1):
        self.token_index -= amount
        self.update_current_token()
        return self.current_token

    def update_current_token(self):
        if self.token_index >= 0 and self.token_index < len(self.tokens):
            self.current_token = self.tokens[self.token_index]

    def parse(self):
        result = self.statements()
        if not result.error and self.current_token.type != TT_END_OF_FILE:
            return result.failure(InvalidSyntaxError(
                self.current_token.position_start, self.current_token.position_end,
                "Expected '+', '-', '*' or '/'"
            ))
        return result

    def if_expr(self):
        result = ParseResult()
        all_cases = result.register(self.if_expr_cases('IF'))
        if result.error:
            return result
        cases, else_cases = all_cases
        return result.success(IfNode(cases, else_cases))

    def if_expr_else(self):
        result = ParseResult()
        else_case = None

        if self.current_token.matches(TT_KEYWORD, 'ELSE'):
            result.register_advancement()
            self.advance()

            if self.current_token.type == TT_NEWLINE:
                result.register_advancement()
                self.advance()

                statements = result.register(self.statements())
                if result.error:
                    return result
                else_case = (statements, True)

                if self.current_token.matches(TT_KEYWORD, 'END'):
                    result.register_advancement()
                    self.advance()
                else:
                    return result.failure(
                        InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end,
                                           "Expected 'END'"))
            else:
                expr = result.register(self.statements())
            if result.error:
                return result
            else_case = (expr, False)

        return result.success(else_case)

    def if_expr_elif_or_else(self):
        result = ParseResult()
        cases, else_case = [], None
        if self.current_token.matches(TT_KEYWORD, 'ELIF'):
            all_cases = result.register(self.if_expr_elif())
            if result.error:
                return result
            cases, else_case = all_cases
        else:
            else_case = result.register(self.if_expr_else())
            if result.error:
                return result
        return result.success((cases, else_case))

    def if_expr_elif(self):
        return self.if_expr_cases('ELIF')

    def if_expr_cases(self, case_keyword):
        result = ParseResult()
        cases = []
        else_case = None

        if not self.current_token.matches(TT_KEYWORD, case_keyword):
            return result.failure(InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end,
                                                     f"Expected '{case_keyword}'"))
        result.register_advancement()
        self.advance()

        condition = result.register(self.expression())
        if result.error:
            return result

        if not self.current_token.matches(TT_KEYWORD, "THEN"):
            return result.failure(InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end,
                                                     f"Expected 'THEN'"))

        result.register_advancement()
        self.advance()

        if self.current_token.type == TT_NEWLINE:
            result.register_advancement()
            self.advance()

            statements = result.register(self.statements())
            if result.error:
                return result
            cases.append((condition, statements, True))

            if self.current_token.matches(TT_KEYWORD, 'END'):
                result.register_advancement()
                self.advance()
            else:
                all_cases = result.register(self.if_expr_elif_or_else())
                if result.error:
                    return result
                new_cases, else_case = all_cases
                cases.extend(new_cases)
        else:
            expr = result.register(self.statement())
            if result.error:
                return result
            cases.append((condition, expr, False))

            all_cases = result.register(self.if_expr_elif_or_else())
            if result.error:
                return result
            new_cases, else_case = all_cases
            cases.extend(new_cases)

        return result.success((cases, else_case))

    def for_expr(self):
        result = ParseResult()

        if not self.current_token.matches(TT_KEYWORD, "FOR"):
            return result.failure(InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end,
                                                     f"Expected 'FOR'"))

        result.register_advancement()
        self.advance()

        if self.current_token.type != TT_IDENTIFIER:
            return result.failure(InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end,
                                                     f"Expected identifier"))

        var_name = self.current_token
        result.register_advancement()
        self.advance()

        if self.current_token.type != TT_EQUALS:
            return result.failure(
                InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end, f"Expected '='"))

        result.register_advancement()
        self.advance()

        start_value = result.register(self.expression())
        if result.error:
            return result

        if not self.current_token.matches(TT_KEYWORD, 'TO'):
            return result.failure(InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end,
                                                     f"Expected 'TO'"))

        result.register_advancement()
        self.advance()

        end_value = result.register(self.expression())
        if result.error:
            return result

        if self.current_token.matches(TT_KEYWORD, "STEP"):
            result.register_advancement()
            self.advance()

            step_value = result.register(self.expression())
            if result.error:
                return result
        else:
            step_value = None

        if not self.current_token.matches(TT_KEYWORD, "THEN"):
            return result.failure(InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end,
                                                     f"Expected 'THEN'"))

        result.register_advancement()
        self.advance()

        if self.current_token.type == TT_NEWLINE:
            result.register_advancement()
            self.advance()

            body = result.register(self.statements())
            if result.error:
                return result

            if not self.current_token.matches(TT_KEYWORD, 'END'):
                return result.failure((InvalidSyntaxError(self.current_token.position_start,
                                                          self.current_token.position_end, f"Expected 'END'")))

            result.register_advancement()
            self.advance()

            return result.success(ForNode(var_name, start_value, end_value, step_value, body, True))

        body = result.register(self.statement())
        if result.error:
            return result

        return result.success(ForNode(var_name, start_value, end_value, step_value, body, False))

    def while_expr(self):
        result = ParseResult()

        if not self.current_token.matches(TT_KEYWORD, "WHILE"):
            return result.failure(InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end,
                                                     f"Expected 'WHILE'"))

        result.register_advancement()
        self.advance()

        condition = result.register(self.expression())
        if result.error:
            return result

        if not self.current_token.matches(TT_KEYWORD, "THEN"):
            return result.failure(InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end,
                                                     f"Expected 'THEN'"))

        result.register_advancement()
        self.advance()

        if self.current_token.type == TT_NEWLINE:
            result.register_advancement()
            self.advance()

            body = result.register(self.statements())
            if result.error:
                return result

            if not self.current_token.matches(TT_KEYWORD, 'END'):
                return result.failure((InvalidSyntaxError(self.current_token.position_start,
                                                          self.current_token.position_end, f"Expected 'END'")))

            result.register_advancement()
            self.advance()

            return result.success(WhileNode(condition, body, True))

        body = result.register(self.statement())
        if result.error:
            return result

        return result.success(WhileNode(condition, body, False))

    def func_def(self):
        result = ParseResult()

        if not self.current_token.matches(TT_KEYWORD, "FUNC"):
            return result.failure(InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end,
                                                     f"Expected 'FUNC'"))
        result.register_advancement()
        self.advance()

        if self.current_token.type == TT_IDENTIFIER:
            var_name_token = self.current_token
            result.register_advancement()
            self.advance()
            if self.current_token.type != TT_LBRACKET:
                return result.failure(
                    InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end,
                                       f"Expected '('"))
        else:
            var_name_token = None
            if self.current_token.type != TT_LBRACKET:
                return result.failure(
                    InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end,
                                       f"Expected identifier or '('"))

        result.register_advancement()
        self.advance()
        arg_name_tokens = []

        if self.current_token.type == TT_IDENTIFIER:
            arg_name_tokens.append(self.current_token)
            result.register_advancement()
            self.advance()

            while self.current_token.type == TT_COMMA:
                result.register_advancement()
                self.advance()

                if self.current_token.type != TT_IDENTIFIER:
                    return result.failure(
                        InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end,
                                           f"Expected identifier"))

                arg_name_tokens.append(self.current_token)
                result.register_advancement()
                self.advance()

            if self.current_token.type != TT_RBRACKET:
                return result.failure(
                    InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end,
                                       f"Expected ',' or ')'"))
        else:
            if self.current_token.type != TT_RBRACKET:
                return result.failure(
                    InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end,
                                       f"Expected identifier or ')'"))

        result.register_advancement()
        self.advance()

        if self.current_token.type == TT_ARROW:
            result.register_advancement()
            self.advance()
            body = result.register(self.expression())
            if result.error:
                return result

            return result.success(FuncDefNode(var_name_token, arg_name_tokens, body, True))

        if self.current_token.type != TT_NEWLINE:
            return result.failure(InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end,
                                                     f"Expected '->' or NEWLINE"))

        result.register_advancement()
        self.advance()

        body = result.register(self.statements())
        if result.error:
            return result

        if not self.current_token.matches(TT_KEYWORD, 'END'):
            return result.failure(InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end,
                                                     f"Expected 'END'"))

        result.register_advancement()
        self.advance()

        return result.success(FuncDefNode(var_name_token, arg_name_tokens, body, False))

    def call(self):
        result = ParseResult()
        atom = result.register(self.atom())
        if result.error:
            return result

        if self.current_token.type == TT_LBRACKET:
            result.register_advancement()
            self.advance()

            arg_nodes = []

            if self.current_token.type == TT_RBRACKET:
                result.register_advancement()
                self.advance()
            else:
                arg_nodes.append(result.register(self.expression()))
                if result.error:
                    return result.failure(
                        InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end,
                                           "Expected ')', 'VAR', 'IF', 'FOR', 'WHILE', 'FUNC', int, float, identifier, '+', '-', '(', '[', or 'NOT'"))

                while self.current_token.type == TT_COMMA:
                    result.register_advancement()
                    self.advance()

                    arg_nodes.append(result.register(self.expression()))
                    if result.error:
                        return result

                if self.current_token.type != TT_RBRACKET:
                    return result.failure(
                        InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end,
                                           f"Expected ',' or ')'"))
                result.register_advancement()
                self.advance()
            return result.success(CallNode(atom, arg_nodes))
        return result.success(atom)

    def atom(self):
        result = ParseResult()
        token = self.current_token

        if token.type in (TT_INT, TT_FLOAT):
            result.register_advancement()
            self.advance()
            return result.success(NumberNode(token))

        if token.type == TT_STRING:
            result.register_advancement()
            self.advance()
            return result.success(StringNode(token))

        elif token.type == TT_IDENTIFIER:
            result.register_advancement()
            self.advance()
            return result.success(VariableAccessNode(token))

        elif token.type == TT_LBRACKET:
            result.register_advancement()
            self.advance()
            expression = result.register(self.expression())
            if result.error:
                return result
            if self.current_token.type == TT_RBRACKET:
                result.register_advancement()
                self.advance()
                return result.success(expression)
            else:
                return result.failure(InvalidSyntaxError(self.current_token.position_start,
                                                         self.current_token.position_end, "Expected ')'"))

        elif token.type == TT_LSQUARE:
            list_expr = result.register(self.list_expr())
            if result.error:
                return result
            return result.success(list_expr)

        elif token.matches(TT_KEYWORD, "IF"):
            if_expr = result.register(self.if_expr())
            if result.error:
                return result
            return result.success(if_expr)

        elif token.matches(TT_KEYWORD, "FOR"):
            for_expr = result.register(self.for_expr())
            if result.error:
                return result
            return result.success(for_expr)

        elif token.matches(TT_KEYWORD, "WHILE"):
            while_expr = result.register(self.while_expr())
            if result.error:
                return result
            return result.success(while_expr)

        elif token.matches(TT_KEYWORD, "FUNC"):
            func_def = result.register(self.func_def())
            if result.error:
                return result
            return result.success(func_def)

        return result.failure((InvalidSyntaxError(token.position_start,
                                                  token.position_end,
                                                  "Expected int, float, identifier, '+', '-', '[', '(', 'IF', 'WHILE, 'FOR', 'FUNC'")))

    def list_expr(self):
        result = ParseResult()
        element_nodes = []
        position_start = self.current_token.position_start.copy()

        if self.current_token.type != TT_LSQUARE:
            return result.failure(
                InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end, f"Expected '['"))

        result.register_advancement()
        self.advance()

        if self.current_token.type == TT_RSQUARE:
            result.register_advancement()
            self.advance()
        else:
            element_nodes.append(result.register(self.expression()))
            if result.error:
                return result.failure(
                    InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end,
                                       "Expected ']', 'VAR', 'IF', 'FOR', 'WHILE', 'FUNC', int, float, identifier, '+', '-', '(', '[' or 'NOT'"))

            while self.current_token.type == TT_COMMA:
                result.register_advancement()
                self.advance()

                element_nodes.append(result.register(self.expression()))
                if result.error:
                    return result

            if self.current_token.type != TT_RSQUARE:
                return result.failure(
                    InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end,
                                       f"Expected ',' or ']'"))
            result.register_advancement()
            self.advance()
        return result.success(ListNode(element_nodes, position_start, self.current_token.position_end.copy()))

    def power(self):
        return self.binary_operation(self.call, (TT_POWER_OF,), self.factor)

    def factor(self):
        result = ParseResult()
        token = self.current_token

        if token.type in (TT_PLUS, TT_MINUS):
            result.register_advancement()
            self.advance()
            factor = result.register(self.factor())
            if result.error:
                return result
            return result.success(UnaryOperationNode(token, factor))

        return self.power()

    def term(self):
        return self.binary_operation(self.factor, (TT_MULTIPLY, TT_DIV))

    def arith_expr(self):
        return self.binary_operation(self.term, (TT_PLUS, TT_MINUS))

    def comp_expr(self):
        result = ParseResult()

        if self.current_token.matches(TT_KEYWORD, "NOT"):
            operator_token = self.current_token
            result.register_advancement()
            self.advance()

            node = result.register(self.comp_expr())
            if result.error:
                return result
            return result.success(UnaryOperationNode(operator_token, node))

        node = result.register(self.binary_operation(self.arith_expr, (TT_EQUALS_EQUALS, TT_NOT_EQUALS, TT_LESS_THAN,
                                                                       TT_GREATER_THAN, TT_LESS_THAN_EQUAL,
                                                                       TT_GREATER_THAN_EQUAL)))

        if result.error:
            return result.failure(InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end,
                                                     "Expected int, float, identifier, '+', '-', '(', '[' or 'NOT'"))
        return result.success(node)

    def statements(self):
        result = ParseResult()
        statements = []
        position_start = self.current_token.position_start.copy()

        while self.current_token.type == TT_NEWLINE:
            result.register_advancement()
            self.advance()
        statement = result.register(self.statement())
        if result.error:
            return result
        statements.append(statement)

        more_statements = True

        while True:
            new_line_count = 0
            while self.current_token.type == TT_NEWLINE:
                result.register_advancement()
                self.advance()
                new_line_count += 1
            if new_line_count == 0:
                more_statements = False

            if not more_statements:
                break

            statement = result.try_register(self.statement())
            if not statement:
                self.reverse(result.to_reverse_count)
                more_statements = False
                continue
            statements.append(statement)
        return result.success(ListNode(statements, position_start, self.current_token.position_end.copy()))

    def statement(self):
        res = ParseResult()
        position_start = self.current_token.position_start.copy()

        if self.current_token.matches(TT_KEYWORD, 'RETURN'):
            res.register_advancement()
            self.advance()

            expr = res.try_register(self.expression())
            if not expr:
                self.reverse(res.to_reverse_count)
            return res.success(ReturnNode(expr, position_start, self.current_token.position_start.copy()))

        if self.current_token.matches(TT_KEYWORD, 'CONTINUE'):
            res.register_advancement()
            self.advance()
            return res.success(ContinueNode(position_start, self.current_token.position_start.copy()))

        if self.current_token.matches(TT_KEYWORD, 'BREAK'):
            res.register_advancement()
            self.advance()
            return res.success(BreakNode(position_start, self.current_token.position_start.copy()))

        expr = res.register(self.expression())
        if res.error:
            return res.failure(InvalidSyntaxError(
                self.current_token.position_start, self.current_token.position_end,
                "Expected 'RETURN', 'CONTINUE', 'BREAK', 'VAR', 'IF', 'FOR', 'WHILE', 'FUN', int, float, identifier, '+', '-', '(', '[' or 'NOT'"
            ))
        return res.success(expr)

    def expression(self):
        result = ParseResult()
        if self.current_token.matches(TT_KEYWORD, "VAR"):
            result.register_advancement()
            self.advance()

            if self.current_token.type != TT_IDENTIFIER:
                return result.failure(
                    InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end,
                                       "Expected identifier"))
            var_name = self.current_token
            result.register_advancement()
            self.advance()

            if self.current_token.type != TT_EQUALS:
                return result.failure(
                    InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end,
                                       "Expected '='"))
            result.register_advancement()
            self.advance()

            expression = result.register(self.expression())
            if result.error:
                return result
            return result.success(VariableAssignNode(var_name, expression))

        node = result.register(self.binary_operation(self.comp_expr, ((TT_KEYWORD, "AND"), (TT_KEYWORD, "OR"))))
        if result.error:
            return result.failure(InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end,
                                                     "Expected 'VAR', 'IF', 'FOR', 'WHILE', 'FUNC', int, float, identifier, '+', '-', '(', '[' or 'NOT'"))
        return result.success(node)

    def binary_operation(self, function_a, operators, function_b=None):
        if function_b == None:
            function_b = function_a
        result = ParseResult()
        left = result.register(function_a())
        if result.error:
            return result

        while self.current_token.type in operators or (self.current_token.type, self.current_token.value) in operators:
            operator_token = self.current_token
            result.register_advancement()
            self.advance()
            right = result.register(function_b())
            if result.error:
                return result
            left = BinaryOperationNode(left, operator_token, right)

        return result.success(left)


# RUNTIME RESULTS

class RunTimeResults:
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = None
        self.error = None
        self.func_return_value = None
        self.loop_should_continue = False
        self.loop_should_break = False

    def register(self, result):
        self.error = result.error
        self.func_return_value = result.func_return_value
        self.loop_should_continue = result.loop_should_continue
        self.loop_should_break = result.loop_should_break
        return result.value

    def success(self, value):
        self.reset()
        self.value = value
        return self

    def success_return(self, value):
        self.reset()
        self.func_return_value = value
        return self

    def success_continue(self):
        self.reset()
        self.loop_should_continue = True
        return self

    def success_break(self):
        self.reset()
        self.loop_should_break = True
        return self

    def failure(self, error):
        self.reset()
        self.error = error
        return self

    def should_return(self):
        return (
                self.error or self.func_return_value or self.loop_should_continue or self.loop_should_break
        )


# VALUES
class Value:
    def __init__(self):
        self.set_position()
        self.set_context()

    def set_position(self, position_start=None, position_end=None):
        self.position_start = position_start
        self.position_end = position_end
        return self

    def set_context(self, context=None):
        self.context = context
        return self

    def added_to(self, other):
        return None, self.illegal_operation(other)

    def subtracted_by(self, other):
        return None, self.illegal_operation(other)

    def multiplied_by(self, other):
        return None, self.illegal_operation(other)

    def divided_by(self, other):
        return None, self.illegal_operation(other)

    def power_of(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_eq(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_ne(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_lt(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_gt(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_lte(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_gte(self, other):
        return None, self.illegal_operation(other)

    def anded_by(self, other):
        return None, self.illegal_operation(other)

    def ored_by(self, other):
        return None, self.illegal_operation(other)

    def notted(self, other):
        return None, self.illegal_operation(other)

    def execute(self, args):
        return RunTimeResults().failure(self.illegal_operation())

    def copy(self):
        raise Exception('No Copy method defined')

    def is_true(self):
        return False

    def illegal_operation(self, other=None):
        if not other:
            other = self
        return RTError(self.position_start, other.position_end, 'Illegal operation', self.context)


class Number(Value):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def added_to(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def subtracted_by(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def multiplied_by(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def divided_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(other.position_start, other.position_end, "Division by zero", self.context)
            return Number(self.value / other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def power_of(self, other):
        if isinstance(other, Number):
            return Number(int(self.value ** other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def get_comparison_eq(self, other):
        if isinstance(other, Number):
            return Number(int(self.value == other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def get_comparison_ne(self, other):
        if isinstance(other, Number):
            return Number(int(self.value != other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def get_comparison_lt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value < other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def get_comparison_gt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value > other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def get_comparison_lte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value <= other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def get_comparison_gte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value >= other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def anded_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value and other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def ored_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value or other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def notted(self, other):
        if isinstance(other, Number):
            return Number(1 if self.value == 0 else 0).set_context(self.context), None

    def copy(self):
        copy = Number(self.value)
        copy.set_position(self.position_start, self.position_end)
        copy.set_context(self.context)
        return copy

    def is_true(self):
        return self.value != 0

    def __repr__(self):
        return str(self.value)


Number.null = Number(0)
Number.false = Number(0)
Number.true = Number(1)


class String(Value):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def added_to(self, other):
        if isinstance(other, String):
            return String(self.value + other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def multiplied_by(self, other):
        if isinstance(other, Number):
            return String(self.value * other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def is_true(self):
        return len(self.value) > 0

    def copy(self):
        copy = String(self.value)
        copy.set_position(self.position_start, self.position_end)
        copy.set_context(self.context)
        return copy

    def __str__(self):
        return self.value

    def __repr__(self):
        return f'"{self.value}"'


class List(Value):
    def __init__(self, elements):
        super().__init__()
        self.elements = elements

    def added_to(self, other):
        new_list = self.copy()
        new_list.elements.append(other)
        return new_list, None

    def subtracted_by(self, other):
        if isinstance(other, Number):
            new_list = self.copy()
            try:
                new_list.elements.pop(other.value)
                return new_list, None
            except IndexError:
                return None, RTError(other.position_start, other.position_end,
                                     "Element at this index could not be removed from this list because index is out of bounds",
                                     self.context)
        else:
            return None, Value.illegal_operation(self, other)

    def divided_by(self, other):
        if isinstance(other, Number):
            try:
                return self.elements[other.value], None
            except IndexError:
                return None, RTError(other.position_start, other.position_end,
                                     "Element at this index could not be retrieved from this list because index is out of bounds",
                                     self.context)
        else:
            return None, Value.illegal_operation(self, other)

    def multiplied_by(self, other):
        if isinstance(other, List):
            new_list = self.copy()
            new_list.elements.extend(other.elements)
            return new_list, None
        else:
            return None, Value.illegal_operation(self, other)

    def copy(self):
        copy = List(self.elements)
        copy.set_position(self.position_start, self.position_end)
        copy.set_context(self.context)
        return copy

    def __str__(self):
        return f'{", ".join([str(element) for element in self.elements])}'

    def __repr__(self):
        return f'[{", ".join([str(element) for element in self.elements])}]'


class BaseFunction(Value):
    def __init__(self, name):
        super().__init__()
        self.name = name or "<anonymous>"

    def generate_new_context(self):
        new_context = Context(self.name, self.context, self.position_start)
        new_context.symbol_table = SymbolTable(new_context.parent.symbol_table)
        return new_context

    def check_args(self, arg_names, args):
        result = RunTimeResults()

        if len(args) > len(arg_names):
            return result.failure(RTError(
                self.position_start, self.position_end,
                f"{len(args) - len(arg_names)} too many args passed into {self}",
                self.context
            ))

        if len(args) < len(arg_names):
            return result.failure(RTError(
                self.position_start, self.position_end,
                f"{len(arg_names) - len(args)} too few args passed into {self}",
                self.context
            ))

        return result.success(None)

    def populate_args(self, arg_names, args, execute_context):
        for i in range(len(args)):
            arg_name = arg_names[i]
            arg_value = args[i]
            arg_value.set_context(execute_context)
            execute_context.symbol_table.set(arg_name, arg_value)

    def check_and_populate_args(self, arg_names, args, execute_context):
        result = RunTimeResults()
        result.register(self.check_args(arg_names, args))
        if result.should_return():
            return result
        self.populate_args(arg_names, args, execute_context)
        return result.success(None)


class Function(BaseFunction):
    def __init__(self, name, body_node, arg_names, should_auto_return):
        super().__init__(name)
        self.body_node = body_node
        self.arg_names = arg_names
        self.should_auto_return = should_auto_return

    def execute(self, args):
        result = RunTimeResults()
        interpreter = Interpreter()
        execute_context = self.generate_new_context()

        result.register(self.check_and_populate_args(self.arg_names, args, execute_context))
        if result.should_return():
            return result

        value = result.register(interpreter.visit(self.body_node, execute_context))
        if result.should_return() and result.func_return_value == None:
            return result

        return_value = (value if self.should_auto_return else None) or result.func_return_value
        return result.success(return_value)

    def copy(self):
        copy = Function(self.name, self.body_node, self.arg_names, self.should_auto_return)
        copy.set_context(self.context)
        copy.set_position(self.position_start, self.position_end)
        return copy

    def __repr__(self):
        return f"<function {self.name}>"


class BuiltInFunction(BaseFunction):
    def __init__(self, name):
        super().__init__(name)

    def execute(self, args):
        result = RunTimeResults()
        execute_context = self.generate_new_context()

        method_name = f'execute_{self.name}'
        method = getattr(self, method_name, self.no_visit_method)

        result.register(self.check_and_populate_args(method.arg_names, args, execute_context))
        if result.should_return():
            return result

        return_value = result.register(method(execute_context))
        if result.should_return():
            return result
        return result.success(return_value)

    def no_visit_method(self, node, context):
        raise Exception(f'No execute_{self.name} method defined')

    def copy(self):
        copy = BuiltInFunction(self.name)
        copy.set_context(self.context)
        copy.set_position(self.position_start, self.position_end)
        return copy

    def __repr__(self):
        return f"<built-in function {self.name}>"

    def execute_print(self, execute_context):
        print(str(execute_context.symbol_table.get('value')))
        return RunTimeResults().success(Number.null)

    execute_print.arg_names = ['value']

    def execute_print_return(self, execute_context):
        return RunTimeResults().success(String(str(execute_context.symbol_table.get('value'))))

    execute_print_return.arg_names = ['value']

    def execute_input(self, execute_context):
        text = input("> ")
        return RunTimeResults().success(String(text))

    execute_input.arg_names = []

    def execute_input_int(self, execute_context):
        while True:
            text = input("> ")
            try:
                number = int(text)
                break
            except ValueError:
                print(f"'{text}' must be an integer. Try again!")
        return RunTimeResults().success(Number(number))

    execute_input_int.arg_names = []

    def execute_clear(self, execute_context):
        os.system('cls' if os.name == 'nt' else 'clear')
        return RunTimeResults().success(Number.null)

    execute_clear.arg_names = []

    def execute_is_number(self, execute_context):
        is_number = isinstance(execute_context.symbol_table.get("value"), Number)
        return RunTimeResults().success(Number.true if is_number else Number.false)

    execute_is_number.arg_names = ["value"]

    def execute_is_string(self, execute_context):
        is_number = isinstance(execute_context.symbol_table.get("value"), String)
        return RunTimeResults().success(Number.true if is_number else Number.false)

    execute_is_string.arg_names = ["value"]

    def execute_is_list(self, execute_context):
        is_number = isinstance(execute_context.symbol_table.get("value"), List)
        return RunTimeResults().success(Number.true if is_number else Number.false)

    execute_is_list.arg_names = ["value"]

    def execute_is_function(self, execute_context):
        is_number = isinstance(execute_context.symbol_table.get("value"), BaseFunction)
        return RunTimeResults().success(Number.true if is_number else Number.false)

    execute_is_function.arg_names = ["value"]

    def execute_append(self, execute_context):
        list_ = execute_context.symbol_table.get("list")
        value = execute_context.symbol_table.get("value")
        if not isinstance(list_, List):
            return RunTimeResults().failure(
                RTError(self.position_start, self.position_end, "First argument must be list", execute_context))

        list_.elements.append(value)
        return RunTimeResults().success(Number.null)

    execute_append.arg_names = ["list", "value"]

    def execute_pop(self, execute_context):
        list_ = execute_context.symbol_table.get("list")
        index = execute_context.symbol_table.get("index")

        if not isinstance(list_, List):
            return RunTimeResults().failure(RTError(
                self.position_start, self.position_end,
                "First argument must be list",
                execute_context
            ))

        if not isinstance(index, Number):
            return RunTimeResults().failure(RTError(
                self.position_start, self.position_end,
                "Second argument must be number",
                execute_context
            ))

        try:
            element = list_.elements.pop(index.value)
        except IndexError:
            return RunTimeResults().failure(RTError(
                self.position_start, self.position_end,
                'Element at this index could not be removed from list because index is out of bounds',
                execute_context
            ))
        return RunTimeResults().success(element)

    execute_pop.arg_names = ["list", "index"]

    def execute_extend(self, execute_context):
        listA = execute_context.symbol_table.get("listA")
        listB = execute_context.symbol_table.get("listB")

        if not isinstance(listA, List):
            return RunTimeResults().failure(RTError(
                self.position_start, self.position_end,
                "First argument must be list",
                execute_context
            ))

        if not isinstance(listB, List):
            return RunTimeResults().failure(RTError(
                self.position_start, self.position_end,
                "Second argument must be list",
                execute_context
            ))

        listA.elements.extend(listB.elements)
        return RunTimeResults().success(Number.null)

    execute_extend.arg_names = ["listA", "listB"]

    def execute_len(self, execute_context):
        list = execute_context.symbol_table.get("list")
        if not isinstance(list, List):
            return RunTimeResults().failure(
                RTError(self.position_start, self.position_end, "Argument must be list", execute_context))

        return RunTimeResults().success(Number(len(list.elements)))

    execute_len.arg_names = ["list"]

    def execute_run(self, execute_context):
        fn = execute_context.symbol_table.get("fn")

        if not isinstance(fn, String):
            return RunTimeResults().failure(RTError(
                self.position_start, self.position_end,
                "Second argument must be string",
                execute_context
            ))

        fn = fn.value

        try:
            with open(fn, "r") as f:
                script = f.read()
        except Exception as e:
            return RunTimeResults().failure(RTError(
                self.position_start, self.position_end,
                f"Failed to load script \"{fn}\"\n" + str(e),
                execute_context
            ))

        _, error = run(fn, script)

        if error:
            return RunTimeResults().failure(RTError(
                self.position_start, self.position_end,
                f"Failed to finish executing script \"{fn}\"\n" +
                error.as_string(),
                execute_context
            ))

        return RunTimeResults().success(Number.null)

    execute_run.arg_names = ["fn"]


BuiltInFunction.print = BuiltInFunction("print")
BuiltInFunction.print_return = BuiltInFunction("print_return")
BuiltInFunction.input = BuiltInFunction("input")
BuiltInFunction.input_int = BuiltInFunction("input_int")
BuiltInFunction.clear = BuiltInFunction("clear")
BuiltInFunction.is_number = BuiltInFunction("is_number")
BuiltInFunction.is_string = BuiltInFunction("is_string")
BuiltInFunction.is_list = BuiltInFunction("is_list")
BuiltInFunction.is_function = BuiltInFunction("is_function")
BuiltInFunction.append = BuiltInFunction("append")
BuiltInFunction.pop = BuiltInFunction("pop")
BuiltInFunction.extend = BuiltInFunction("extend")
BuiltInFunction.len = BuiltInFunction("len")
BuiltInFunction.run = BuiltInFunction("run")


# CONTEXT

class Context:
    def __init__(self, display_name, parent=None, parent_entry_position=None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_position = parent_entry_position
        self.symbol_table = None


# SYMBOL TABLE

class SymbolTable:
    def __init__(self, parent=None):
        self.symbols = {}
        self.parent = parent

    def get(self, name):
        value = self.symbols.get(name, None)
        if value == None and self.parent:
            return self.parent.get(name)
        return value

    def set(self, name, value):
        self.symbols[name] = value

    def remove(self, name):
        del self.symbols[name]


# INTERPRETER:

class Interpreter:
    def visit(self, node, context):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)

    def no_visit_method(self, node, context):
        raise Exception(f'No visit_{type(node).__name__} method defined')

    def visit_StringNode(self, node, context):
        return RunTimeResults().success(
            String(node.token.value).set_context(context).set_position(node.position_start, node.position_end))

    def visit_NumberNode(self, node, context):
        return RunTimeResults().success(
            Number(node.token.value).set_context(context).set_position(node.position_start, node.position_end))

    def visit_ListNode(self, node, context):
        result = RunTimeResults()
        elements = []

        for element_node in node.element_node:
            elements.append(result.register(self.visit(element_node, context)))
            if result.should_return():
                return result
        return result.success(List(elements).set_context(context).set_position(node.position_start, node.position_end))

    def visit_VariableAccessNode(self, node, context):
        result = RunTimeResults()
        var_name = node.var_name_token.value
        value = context.symbol_table.get(var_name)

        if not value:
            return result.failure(
                RTError(node.position_start, node.position_end, f"'{var_name}' is not defined", context))
        value = value.copy().set_position(node.position_start, node.position_end).set_context(context)
        return result.success(value)

    def visit_VariableAssignNode(self, node, context):
        result = RunTimeResults()
        var_name = node.var_name_token.value
        value = result.register(self.visit(node.value_node, context))
        if result.should_return():
            return result

        context.symbol_table.set(var_name, value)
        return result.success(value)

    def visit_BinaryOperationNode(self, node, context):
        run_time_result = RunTimeResults()
        left = run_time_result.register(self.visit(node.left_node, context))
        if run_time_result.error:
            return run_time_result
        right = run_time_result.register(self.visit(node.right_node, context))
        if run_time_result.error:
            return run_time_result

        if node.operator_token.type == TT_PLUS:
            result, error = left.added_to(right)
        elif node.operator_token.type == TT_MINUS:
            result, error = left.subtracted_by(right)
        elif node.operator_token.type == TT_MULTIPLY:
            result, error = left.multiplied_by(right)
        elif node.operator_token.type == TT_DIV:
            result, error = left.divided_by(right)
        elif node.operator_token.type == TT_POWER_OF:
            result, error = left.power_of(right)
        elif node.operator_token.type == TT_EQUALS_EQUALS:
            result, error = left.get_comparison_eq(right)
        elif node.operator_token.type == TT_NOT_EQUALS:
            result, error = left.get_comparison_ne(right)
        elif node.operator_token.type == TT_LESS_THAN:
            result, error = left.get_comparison_lt(right)
        elif node.operator_token.type == TT_GREATER_THAN:
            result, error = left.get_comparison_gt(right)
        elif node.operator_token.type == TT_LESS_THAN_EQUAL:
            result, error = left.get_comparison_lte(right)
        elif node.operator_token.type == TT_GREATER_THAN_EQUAL:
            result, error = left.get_comparison_gte(right)
        elif node.operator_token.matches(TT_KEYWORD, "AND"):
            result, error = left.anded_by(right)
        elif node.operator_token.matches(TT_KEYWORD, "OR"):
            result, error = left.ored_by(right)

        if error:
            return run_time_result.failure(error)
        else:
            return run_time_result.success(result.set_position(node.position_start, node.position_end))

    def visit_UnaryOperationNode(self, node, context):
        run_time_result = RunTimeResults()
        number = run_time_result.register(self.visit(node.node, context))
        if run_time_result.error:
            return run_time_result

        error = None

        if node.operator_token.type == TT_MINUS:
            number, error = number.multiplied_by(Number(-1))
        elif node.operator_token.matches(TT_KEYWORD, "NOT"):
            number, error = number.notted()

        if error:
            run_time_result.failure(error)
        else:
            return run_time_result.success(number.set_position(node.position_start, node.position_end))

    def visit_IfNode(self, node, context):
        result = RunTimeResults()

        for condition, expr, should_return_null in node.cases:
            condition_value = result.register(self.visit(condition, context))
            if result.should_return():
                return result

            if condition_value.is_true():
                expr_value = result.register(self.visit(expr, context))
                if result.should_return():
                    return result
                return result.success(Number.null if should_return_null else expr_value)

        if node.else_case:
            expr, should_return_null = node.else_case
            else_value = result.register(self.visit(expr, context))
            if result.should_return():
                return result
            return result.success(Number.null if should_return_null else else_value)

        return result.success(Number.null)

    def visit_ForNode(self, node, context):
        result = RunTimeResults()
        elements = []

        start_value = result.register(self.visit(node.start_value_node, context))
        if result.should_return():
            return result

        end_value = result.register(self.visit(node.end_value_node, context))
        if result.should_return():
            return result

        if node.step_value_node:
            step_value = result.register(self.visit(node.step_value_node, context))
            if result.should_return():
                return result
        else:
            step_value = Number(1)

        i = start_value.value

        if step_value.value >= 0:
            condition = lambda: i < end_value.value
        else:
            condition = lambda: i > end_value.value

        while condition():
            context.symbol_table.set(node.var_name_token.value, Number(i))
            i += step_value.value

            value = result.register(self.visit(node.body_node, context))
            if result.should_return() and result.loop_should_continue == False and result.loop_should_break == False:
                return result

            if result.loop_should_continue:
                continue

            if result.loop_should_break:
                break

            elements.append(value)

        return result.success(Number.null if node.should_return_null else
                              List(elements).set_context(context).set_position(node.position_start, node.position_end))

    def visit_WhileNode(self, node, context):
        result = RunTimeResults()
        elements = []
        while True:
            condition = result.register(self.visit(node.condition_node, context))
            if result.should_return():
                return result

            if not condition.is_true():
                break

            value = result.register(self.visit(node.body_node, context))
            if result.should_return() and result.loop_should_continue == False and result.loop_should_break == False:
                return result

            if result.loop_should_continue:
                continue

            if result.loop_should_break:
                break

            elements.append(value)

        return result.success(Number.null if node.should_return_null else
                              List(elements).set_context(context).set_position(node.position_start, node.position_end))

    def visit_FuncDefNode(self, node, context):
        result = RunTimeResults()
        func_name = node.var_name_token.value if node.var_name_token else None
        body_node = node.body_node
        arg_names = [arg_name.value for arg_name in node.arg_name_tokens]

        func_value = Function(func_name, body_node, arg_names, node.should_auto_return).set_context(
            context).set_position(node.position_start,
                                  node.position_end)
        if node.var_name_token:
            context.symbol_table.set(func_name, func_value)

        return result.success(func_value)

    def visit_CallNode(self, node, context):
        result = RunTimeResults()
        args = []

        value_to_call = result.register(self.visit(node.node_to_call, context))

        if result.should_return():
            return result

        value_to_call = value_to_call.copy().set_position(node.position_start, node.position_end)

        for arg_node in node.arg_nodes:
            args.append(result.register(self.visit(arg_node, context)))
            if result.should_return():
                return result

        return_value = result.register(value_to_call.execute(args))
        if result.should_return():
            return result
        return_value = return_value.copy().set_position(node.position_start, node.position_end).set_context(context)
        return result.success(return_value)

    def visit_ReturnNode(self, node, context):
        result = RunTimeResults()
        if node.node_to_return:
            value = result.register(self.visit(node.node_to_return, context))
            if result.should_return():
                return result
        else:
            value = Number.null

        return result.success_return(value)

    def visit_ContinueNode(self, node, context):
        return RunTimeResults().success_continue()

    def visit_BreakNode(self, node, context):
        return RunTimeResults().success_break()


# RUN

global_symbol_table = SymbolTable()
global_symbol_table.set("NULL", Number.null)
global_symbol_table.set("TRUE", Number.false)
global_symbol_table.set("FALSE", Number.true)
global_symbol_table.set("PRINT", BuiltInFunction.print)
global_symbol_table.set("PRINT_RETURN", BuiltInFunction.print_return)
global_symbol_table.set("INPUT", BuiltInFunction.input)
global_symbol_table.set("INPUT_INT", BuiltInFunction.input_int)
global_symbol_table.set("CLEAR", BuiltInFunction.clear)
global_symbol_table.set("CLS", BuiltInFunction.clear)
global_symbol_table.set("IS_NUM", BuiltInFunction.is_number)
global_symbol_table.set("IS_STR", BuiltInFunction.is_string)
global_symbol_table.set("IS_LIST", BuiltInFunction.is_list)
global_symbol_table.set("IS_FUNC", BuiltInFunction.is_function)
global_symbol_table.set("APPEND", BuiltInFunction.append)
global_symbol_table.set("POP", BuiltInFunction.pop)
global_symbol_table.set("EXTEND", BuiltInFunction.extend)
global_symbol_table.set("LEN", BuiltInFunction.len)
global_symbol_table.set("RUN", BuiltInFunction.run)


def run(filename, text):
    # Generate tokens
    lexer = Lexer(filename, text)
    tokens, error = lexer.make_tokens()
    if error:
        return None, error

    # Generate AST
    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error:
        return None, ast.error

    # Run program
    interpreter = Interpreter()
    context = Context('<program>')
    context.symbol_table = global_symbol_table
    result = interpreter.visit(ast.node, context)

    return result.value, result.error
