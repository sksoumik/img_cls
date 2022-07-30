def get_html_table(image_paths, names, column_labels):
    s = '<table align="center">'
    if column_labels:
        s += (
            '<tr><th><h4 style="font-family:Arial">'
            + column_labels[0]
            + '</h4></th><th><h4 style="font-family:Arial">'
            + column_labels[1]
            + "</h4></th></tr>"
        )

    for name, image_path in zip(names, image_paths):
        s += '<tr><td><img height="80" src="/' + image_path + '" ></td>'
        s += '<td style="text-align:center">' + name + "</td></tr>"
    s += "</table>"

    return s


head_html = """
<head>
<title>Satellite Scene detection</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
</head>

<body style="background-color: #ebebeb;">
<center>
"""

body_html = """
        <h4 style="text-align: center;">
        Upload a satellite image to predict the scene type
        </h4>

        """
