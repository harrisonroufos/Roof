import roof

while True:
    text = input('roof > ')
    result, error = roof.run(text, "<file_example>")

    if error:
        print(error.as_string())
    elif result:
        print(repr(result))
