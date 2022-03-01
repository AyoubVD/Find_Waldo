import shutil
import os
    
source_dir = 'C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/training/notwaldo'
target_dir = 'C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/testing/notwaldo'

source_dir2 = 'C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/training/waldo'
target_dir2 = 'C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/testing/waldo'
    
file_names = os.listdir(source_dir)

for i in range (1300):
    shutil.move(os.path.join(source_dir, file_names[i]), target_dir)
    
file_names = os.listdir(source_dir2)
for i in range (9):
    shutil.move(os.path.join(source_dir2, file_names[i]), target_dir2)