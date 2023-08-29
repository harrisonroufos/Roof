import roof

while True:
    text = input('roof > ')
    if text.strip() == "":
        continue
    result, error = roof.run('<file>', text)

    if error:
        print(error.as_string())
    elif result:
        if len(result.elements) == 1:
            print(repr(result.elements[0]))
        else:
            print(repr(result))
