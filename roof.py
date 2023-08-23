"""
TOKENS
LEXER
"""
import resource

from string_with_arrows import *
import string

# CONSTANTS

DIGITS = "0123456789"
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS


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
        super().__init__(position_start, position_end, "Invalid Syntax", details)


class RTError(Error):
    def __init__(self, position_start, position_end, details, context):
        super().__init__(position_start, position_end, "Runtime Error", details)
        self.context = context


class ExpectedCharError:
    def __init__(self, position_start, position_end, details):
        super().__init__(position_start, position_end, "Expected Character", details)

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

KEYWORDS = ["VAR", "AND", "OR", "NOT", "IF", "THEN", "ELIF", "ELSE"]


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
            elif self.current_char in LETTERS:
                tokens.append(self.make_identifier())
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
            elif self.current_char == "^":
                tokens.append((Token(TT_POWER_OF, position_start=self.position)))
                self.advance()
            # elif self.current_char == "=":
            #     tokens.append((Token(TT_EQUALS, position_start=self.position)))
            #     self.advance()
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


# NODES

class NumberNode:
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
        self.position_end = (self.else_case or self.cases[len(self.cases) - 1][0]).position_end


# PARSER RESULTS

class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
        self.advance_count = 0

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
            return result.failure(InvalidSyntaxError(
                self.current_token.position_start, self.current_token.position_end,
                "Expected '+', '-', '*' or '/'"
            ))
        return result

    def if_expr(self):
        result = ParseResult()
        cases = []
        else_case = None

        if not self.current_token.matches(TT_KEYWORD, "IF"):
            return result.failure(InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end,
                                                     f"Expected 'IF'"))

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

        expr = result.register(self.expression())
        if result.error:
            return result
        cases.append((condition, expr))

        while self.current_token.matches(TT_KEYWORD, "ELIF"):
            result.register_advancement()
            self.advance()

            condition = result.register(self.expression())
            if result.error:
                return result

            if not self.current_token.matches(TT_KEYWORD, "THEN"):
                return result.failure(
                    InvalidSyntaxError(self.current_token.position_start, self.current_token.position_end,
                                       f"Expected 'THEN'"))

            result.register_advancement()
            self.advance()

            expr = result.register(self.expression())
            if result.error:
                return result
            cases.append((condition, expr))

        if self.current_token.matches(TT_KEYWORD, "ELSE"):
            result.register_advancement()
            self.advance()

            else_case = result.register(self.expression())
            if result.error:
                return result

        return result.success(IfNode(cases, else_case))

    def atom(self):
        result = ParseResult()
        token = self.current_token

        if token.type in (TT_INT, TT_FLOAT):
            result.register_advancement()
            self.advance()
            return result.success(NumberNode(token))

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

        elif token.matches(TT_KEYWORD, "IF"):
            if_expr = result.register(self.if_expr())
            if result.error:
                return result
            return result.success(if_expr)

        return result.failure((InvalidSyntaxError(token.position_start,
                                                  token.position_end,
                                                  "Expected int, float, identifier, '+', '-' or '('")))

    def power(self):
        return self.binary_operation(self.atom, (TT_POWER_OF,), self.factor)

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
                                                     "Expected int, float, identifier, '+', '-', '(', 'NOT'"))
        return result.success(node)

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
                                                     "Expected 'VAR', int, float, identifier, '+', '-' or '('"))
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
        self.value = None
        self.error = None

    def register(self, result):
        if result.error:
            self.error = result.error
        return result.value

    def success(self, value):
        self.value = value
        return self

    def failure(self, error):
        self.error = error
        return self


# VALUES
class Number:
    def __init__(self, value):
        self.value = value
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
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.context), None

    def subtracted_by(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None

    def multiplied_by(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context), None

    def divided_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(other.position_start, other.position_end, "Division by zero", self.context)
            return Number(self.value / other.value).set_context(self.context), None

    def power_of(self, other):
        if isinstance(other, Number):
            return Number(int(self.value ** other.value)).set_context(self.context), None

    def get_comparison_eq(self, other):
        if isinstance(other, Number):
            return Number(int(self.value == other.value)).set_context(self.context), None

    def get_comparison_ne(self, other):
        if isinstance(other, Number):
            return Number(int(self.value != other.value)).set_context(self.context), None

    def get_comparison_lt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value < other.value)).set_context(self.context), None

    def get_comparison_gt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value > other.value)).set_context(self.context), None

    def get_comparison_lte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value <= other.value)).set_context(self.context), None

    def get_comparison_gte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value >= other.value)).set_context(self.context), None

    def anded_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value and other.value)).set_context(self.context), None

    def ored_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value or other.value)).set_context(self.context), None

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


# CONTEXT

class Context:
    def __init__(self, display_name, parent=None, parent_entry_position=None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_position = parent_entry_position
        self.symbol_table = None


# SYMBOL TABLE

class SymbolTable:
    def __init__(self):
        self.symbols = {}
        self.parent = None

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

    def visit_NumberNode(self, node, context):
        return RunTimeResults().success(
            Number(node.token.value).set_context(context).set_position(node.position_start, node.position_end))

    def visit_VariableAccessNode(self, node, context):
        result = RunTimeResults()
        var_name = node.var_name_token.value
        value = context.symbol_table.get(var_name)

        if not value:
            return result.failure(
                RTError(node.position_start, node.position_end, f"'{var_name}' is not defined", context))
        value = value.copy().set_position(node.position_start, node.position_end)
        return result.success(value)

    def visit_VariableAssignNode(self, node, context):
        result = RunTimeResults()
        var_name = node.var_name_token.value
        value = result.register(self.visit(node.value_node, context))
        if result.error:
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

        for condition, expr in node.cases:
            condition_value = result.register(self.visit(condition, context))
            if result.error:
                return result

            if condition_value.is_true():
                expr_value = result.register(self.visit(expr, context))
                if result.error:
                    return result
                return result.success(expr_value)

        if node.else_case:
            else_value = result.register(self.visit(node.else_case, context))
            if result.error:
                return result
            return result.success(else_value)

        return result.success(None)


# RUN

global_symbol_table = SymbolTable()
global_symbol_table.set("NULL", Number(0))
global_symbol_table.set("TRUE", Number(1))
global_symbol_table.set("FALSE", Number(0))


def run(file_name, text):
    # Generate tokens
    lexer = Lexer(file_name, text)
    tokens, error = lexer.make_tokens()
    if error:
        return None, error

    # Generate AST = abstract syntax tree
    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error:
        return None, ast.error

    # Run program
    interpreter = Interpreter()
    context = Context("<program>")
    context.symbol_table = global_symbol_table
    result = interpreter.visit(ast.node, context)

    return result.value, result.error
