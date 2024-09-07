import os
import json

pkl_files = []
json_files = []
common_files = []

# Variables con las rutas a los directorios donde se encuentran los archivos .pkl y .json
folder = 'wprime_plus_b/outs/ttbar/1b1mu1tau/mu/2017'
folder_metadata = 'wprime_plus_b/outs/ttbar/1b1mu1tau/mu/2017/metadata'

# Para los archivos .pkl en el directorio especificado
for file in os.listdir(folder):
    if file.endswith('.pkl'):
        file_path = os.path.join(folder, file)
        file_size = os.path.getsize(file_path)
        if file_size < 100:  # 1 kilobyte = 1024 bytes
            pkl_files.append(os.path.splitext(file)[0])

# Para los archivos .json en el subdirectorio 'metadata'
for file in os.listdir(folder_metadata):
    if file.endswith('.json'):
        file_path = os.path.join(folder_metadata, file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            if 'raw_final_nevents' in data and data['raw_final_nevents'] != 0:
                file_name = os.path.splitext(file)[0]
                file_name = file_name.replace('_metadata', '')  # Eliminar '_metadata' del nombre del archivo
                json_files.append(file_name)

# Para encontrar archivos con el mismo nombre
for file in pkl_files:
    if file in json_files:
        common_files.append(file)

# En caso de que la lista sea vacia, regresa un mensaje en inglés confirmando que no hay archivos zombies
if not common_files:
    print('No zombie files found.')