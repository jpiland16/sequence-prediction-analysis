import os

def save_html(data, save_dir):
    string, depth = create_json_string(1, data)
    with open("summary.html", 'r') as file:
        template = file.read()
    template = template.replace("_DEPTH_", str(depth))
    template = template.replace("_IMAGE_DATA_", string)

    with open(os.path.join(save_dir, "summary-res.html"), 'w') as file:
        file.write(template)

def create_json_string(depth, data, full_name=""):
    if depth > 3: 
        raise ValueError("Depth greater than 3 is not currently supported " + 
        "for making HTML files. (Your data was still saved.)")
    else:
        if type(data["data_set"][0]["data_point"]) == list:
            if full_name == "":
                full_name = "graph"
            return ("""        {
                        'data_set': '""" + full_name + """.png', 
                        'step_name': '""" + data["step_name"] + """'
                    }""", depth)

        if full_name != "":
            full_name += " "

        data_points = [create_json_string(depth + 1, 
                data["data_set"][i]["data_point"], 
                full_name + data["step_name"] 
                    + "_" + str(data["data_set"][i]["step_value"])
            )
            for i in range(len(data["data_set"]))]

        data_set_text = ""

        for i, data_point in enumerate(data_points):
            # The [0] below is for ignoring the depth calculation
            data_set_text += """    {
                    'data_point': """ + data_point[0] + """, 
                    'step_value': """ + str(data["data_set"][i]["step_value"]) + """
                },
            """

        calculated_depth = data_points[0][1]
    
        return (""" {
                'data_set': [""" + data_set_text + """], 
                'step_name': 'num-of-bands'
            }
            """, calculated_depth)