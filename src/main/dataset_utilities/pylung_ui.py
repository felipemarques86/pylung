import configparser

import PySimpleGUI as sg

from main.utilities.utilities_lib import display_original_image_bbox

config = configparser.ConfigParser()

# check config.ini is in the root folder
#config_file = config.read('config.ini')
#if len(config_file) == 0:
#    raise Exception("config.ini file not found")
#    directory = config['DATASET'][f'processed_{dataset_type}_location']
#    dataset_reader = CustomLidcDatasetReader(location=directory + f'/{dataset_name}/')
#    dataset_reader.load_custom()

def display_dataset_images(dataset_reader):

    # First the window layout in 2 columns

    file_list_column = [
        [
            sg.Listbox(
                values=range(0, len(dataset_reader.images)), enable_events=True, size=(40, 20), key="-IMAGE LIST-"
            )
        ],
    ]

    # For now will only show the name of the file that was chosen
    image_viewer_column = [
        [sg.Text("Choose an image from list on left:")],
        [sg.Text(size=(40, 1), key="-TOUT-")],
    ]

    # ----- Full layout -----
    layout = [
        [
            sg.Column(file_list_column),
            #sg.VSeperator(),
            #sg.Column(image_viewer_column),
        ]
    ]

    window = sg.Window("Image Viewer", layout)

    # Run the Event Loop
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        # Folder name was filled in, make a list of files in the folder
        if event == "-IMAGE LIST-":  # A file was chosen from the listbox
            print(event)
            print(values)
            display_original_image_bbox(dataset_reader.images[values['-IMAGE LIST-'][0]], dataset_reader.annotations[values['-IMAGE LIST-'][0]])
            # try:
            #     filename = os.path.join(
            #         values["-FOLDER-"], values["-FILE LIST-"][0]
            #     )
            #     window["-TOUT-"].update(filename)
            #     window["-IMAGE-"].update(filename=filename)
            #
            # except:
            #     pass

    window.close()
