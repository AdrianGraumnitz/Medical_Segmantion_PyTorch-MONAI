Projektantrag
Risikoanalyse

Legende:
	- "" Sätze für die Doku
	- ?? Änderungen im Code welche zu unvorhergesehenen Folgen führen könnte -> genau beobachten
	- !! Überarbeiten/ tiefergehend mit auseinandersetzen
	- $$ Nicht implementierte Ergänzungen/Ideen
	
-------------------------------
-------------------Programm--------------------------------

------genutzte Software-----
Windows 10/11
Word
Anaconda
VS Code
Notepad++
Jupyter Notebook

Virtuelle Umgebung in Anaconda programmiert:
	Installierte Libraries:
		-monai
		-Pytorch
		-nibabel
		-torchinfo
		-torchmetrics
		-torchvision
		-typing
		-tqdm
		-dicom2nifti
	
	
------Preprocess------
Aktuelles verzeichnis zeigen : Path(__file__).parent

Funktionen:
	-create_groups: patient_name = gibt die pfadnamen (Dateinamen) zurück.
					numberfolders = teilt den inhalt eines underverzeichnisses durch die variabel Number_Slices.
					Ich würde gern ein Feedback bekommen wenn der Pfad schon existiert, klappt noch nicht.
""	Anstatt wie in vielen beispielen das os modul zu nutzen um durch verzeichnise zu navigieren habe ich mit für die objektorientierte und modernere Version des pathlib moduls entschieden.
	-dicom2nifti: Durch die mit Dicoms gefüllten vorher erstellten Patientenordner wird durchiteriert und es werden niftis für jeden ordner erstellt.
	-find empty: Niftis werden mit Nibabel geladen.
				 get_fdata = gibt ein numpy array mit den voxelwerten des Volumens zurück.
				 len(unique) > 2 = zählt die eindeutigen werte und gibt True zurück wenn es mehr als zwei sind.
??				 Habe die Funktion mit .unlink() ergänzt, sollte alle leeren Bilder löschen.
	-set_seed: Zufallswerte für berechnung auf cpu/gpu festlegen.
!!	-prepare: über parameter informieren, sind die Werte gut?
!!			  Die Pfade könnten Probleme verursachen.
			  Dictionaries werden mit list comprehension und der zip funktion erstellt. Zip Funktion erstellt einen iterator von paarweisen Tuples.
			  transforms: Überlegen ob die train_transforms überarbeitet werden können/sollen/müssen.
						  LoadImaged: Läd Bild datei und konvertiert sie zu einem Tensor.
						  Spacingd:	pixdim = Skaliert die größe der Voxel und ihr abstand untereinander gemessen vom Voxelzentrum.
									interpolation = Schätz die Anordnung der Pixel auf den Voxeln.
									bilinear (Wird für die Volumes genutzt) = berechnet/schätzt Pixelwerte in dem es den Mittelwert der 4 (2 Dimensionaler Raum)/ 8 (3 Dimensionaler Raum) nächsten Punkte berechnet.
									nearest (Wird für die Segmente genutz) = Der nächstgelegenen Punkt/Pixel wird verwendet umd den Zielpunkt zu schätzen.
						  Orientationd: Legt mit RAS eine Konvention zur ausrichtung der Achsen fest, wichtig für konsistente Daten.
						  CropForgroundsd: Schneidet alle Werte = 0 Weg bis zum Rand der Voxel welche > 0 sind.
						  ScaleIntensityRanged: Manipuliert die Kontrastwerte.
												a_min, a_max = definiert ursprüngliche Intervallbereiche der Pixel. Pixelwerte außerhalb dieses Bereiches werden weggeschnitten.
												b_min, b_max = Zielintervallbereich in welchen die Pixel transformiert werden sollen.
												clip = Wenn auf True gesetzt werden die Werte die sich außerhalb des Zielintervalls befinden auf b_min oder b_max gesetzt.
						  Resize: spatial_size: Anzahl der Voxel.
$$						  Einen Normalizer verwenden (z.b. transforms.NormalizeIntensity(keys = ['vol'], non_zero = True)

---------U-Net------------
	Parameter:
		spatial_dims: Anzahl der Bilddimensionen (3-D Bild)
		in_channels: Input Knoten (Features)
		out_channels: Output Knoten (Features)
		channels: Wieviele Features jeweils in die Layerstacks gesteckt werden
		strides: Größe der Schritte der Faltungschichten
		num_res_units: Layerstacks mit shortcut verbindungen, praktisch da sie nicht durch jede schicht backpropagieren müssen (Praktisch für Netze mit sehr vielen Layern -> Deep Nets)
		norm = welche Form der Normalisation zwischen den Schichten stattfindet. In dem Fall wird eine Batchnormalisierung auf die Aktivierungswerte bevor sie in die Aktivierungsfunktion gehen angewand.
$$		Die Layer des U-Nets lassen sich einfrieren, dies könnte für ein vortrainiertes Unet von nutzen sein (Falls die Layerstacks keine Eigennamen besitzen mit dem Index ansprechen)
		









	
label = label != 0 nur binäre segmentation in mutlisegmentationen wird das label nach verwendung des softmaxes auf die prediction direkt mit der prediction verglichen
Softmax nur im Training verwenden