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
<title>Natural Scene Classification</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
</head>

<body style="background-color: #ebebeb;">
<center>
"""

body_html = """
        <h2 style="text-align: center;">
        Upload one or multiple natural scene images to predict the scene type
        </h2>

        """

pred_html = """
        <h2 style="text-align: center; color: red;">
        Predictions for the images you uploaded
        </h2>
"""

upload_file_html = """
    <br/>
    <br/>
    <form  action="/uploadfiles/" enctype="multipart/form-data" method="post">
    <input name="files" type="file" multiple>
    <input type="submit">
    </form>
    </body>
    """
