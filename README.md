Projektantrag  
Risikoanalyse  

Legende:  
	- "" Sätze für die Doku  
	- ?? Änderungen im Code welche zu unvorhergesehenen Folgen führen könnte -> genau beobachten  
	- !! Überarbeiten/ tiefergehend mit auseinandersetzen  
	- $$ Nicht implementierte Ergänzungen/Ideen  
	- ++ Gelöst  
	
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
		-torchmetrics (brauch ich vielleicht gar nicht (nutze dice_metric))
		-torchvision 
		-typing (deprecated) 
		-tqdm  
!!		-nii2dcm (um die anwendung konsistent in python zu haben)
		-dicom2nifti
		-tensorboard
		-icecream (debuggen)
		-mlxtend (plotting confusion matrix)
		-open3d (erstellt ein Mesh der predictions)
		-scikit image/skimage (zum plotten von meshes)
		-numpy-stl (Um mesh Daten als .stl Datei Abzuspeichern.
	
------data_preprocess--------  
	Bereitet den Datensatz für den Dataloader vor und Unterstützt die Anwendenden Personen.  
------Preprocess------  
Aktuelles verzeichnis zeigen : Path(__file__).parent  

Funktionen:  
	-create_groups: patient_name = gibt die pfadnamen (Dateinamen) zurück.  
					numberfolders = teilt den inhalt eines underverzeichnisses durch die variabel Number_Slices.  
					Ich würde gern ein Feedback bekommen wenn der Pfad schon existiert, klappt noch nicht.  
					Wichtig: der Pfad muss auf das Train/Test directory zeigen, nicht auf die unterordner ('test_data' / 'dicom' /'images_train').  
					Dateien müssen vorher in unterordner (z.B.: "heart_0") gesteckt werden aber die Funktion greift auf den überordner ("train") zu
""	Anstatt wie in vielen Beispielen das os Modul zu nutzen um durch Verzeichnise zu navigieren habe ich mich für die objektorientierte und modernere Version des pathlib Moduls entschieden.  
	-dicom2nifti: Durch die mit Dicoms gefüllten vorher erstellten Patientenordner wird durchiteriert und es werden niftis für jeden ordner erstellt.  
	-find empty: Niftis werden mit Nibabel geladen.  
				 get_fdata = gibt ein numpy array mit den voxelwerten des Volumens zurück.  
				 len(unique) > 2 = zählt die eindeutigen werte und gibt True zurück wenn es mehr als zwei sind.  
??				 Habe die Funktion mit .unlink() ergänzt, sollte alle leeren Bilder löschen.  
	-set_seed: Zufallswerte für berechnung auf cpu/gpu festlegen. 
	-edit_label: Funktion erkennt die Anzahl der unique Werte (Grauwerte) und maped sie zu einem Index in ein Dictionary. Löscht die Originaldatei wenn gemapped Datei existiert  
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
!!						  Resize: spatial_size: Anzahl der Voxel. Es besteht die Möglichkeit, dass Spacingd und Resized miteinander interferieren, da Spacingd die Größe der Voxel skaliert, während Resized die Anzahl der Voxel verändert  
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
$$		Torchinfo summary einbauen um der User*in eine bessere Visualisierung des Models zu liefern  
  
--------engine-------------    
	Funktionen:  
		dice_metric: Berechent eine Metrik um die genauigkeit des Models zu messen.  
!!++				 Liegen die Labels im one-hot-Format vor? bedeutet liegen sie in Klassenvektoren in ganzzahlen vor? (wichtig für DiceLoss parameter to_onehot_y = True/False)  
					 Durch die pytorch Tensor Struktur liegen die Bilder [1, 1, 128, 128, 128] schon in der one hot codierung vor. Jeder Wert in diesem Tensor steht für die Klasse des Pixels an dieser Position im Bild.  
					 Das bedeutet, dass die Werte in diesem Tensor direkt als Klassenlabels für jedes Pixel im Bild interpretiert werden können.  
					 Beispiel für eine one hot codierung: Klasse A = [1, 0, 0]  
														  Klasse B = [0, 1, 0]  
														  Klasse C = [0, 0, 1] zusammengefasster Vektor: [1, 1, 1]  
					 Squared_pred: wenn True werden die logits oder predictions quadriert um die Unterschiede zwischen den predictions und labels zu verstärken  
					 Der dice_value wird von 1 abgezogen so das eine Metric entsteht bei der die 1 das bestmögliche Ergbniss ist.  
$$++	train_step: Model muss noch gespeichert werden. Gibt train loss und train metrik zurück  
		test_step: Gibt test loss und test metrik zurück  
!!		train: save_model Funktion integriert. Optisch hervorgehoben, später besser integrieren  
			   Habe ich mit Summary writer ergänzt  
$$		calculate weights (nur für cross entropy loss): Berechnet die Gewichtung für die Loss Funktion basierend auf der relativen häufigkeit der Klasse. Soll ein mögliches Klassenungleichgewicht in den Daten ausgleichen  
		perform_inference: Funktion welchen einen prediction Datensatz zurück gibt.  
  
---------utils-------------  
!!	Die target_dir Variabel ist nicht Konsistent als Datentyp (manchmal String manchmal pathlib Objekt)  
	Funktionen:  
		save_model: Übernimmt als Parameter das Model und ein Zielverzeichnis.  
					Assert Anweisung ist zur Absicherung des Models mit richtiger Dateiendung vorgesehen. Falls weder .pth oder .pt asl endung genutzt wurden wird eine Exception geworfen, welche auf den Fehler mit einer Message hinweist.  
					Falls das Zielverzeichnis nicht gefunden wird, wird es erstellt  
!!					Mir ist es nicht gelungen den Modell Parameter als Monai zu Annotieren, habe PyTorch genommen  
		load_weights: Übernimmt als Parameter das Modell und das Verzeichnis unter dem die Gewichte gespeichert sind.  
					  Gibt ein Feedback ob das Modell none ist oder nicht  
!!					  Mir ist es nicht gelungen den Modell Parameter als Monai zu Annotieren, habe PyTorch genommen.  
		save_metric: Speichert das durchschnittliche loss und die durchschnittliche dice metric in einer jeweiligen numpy Datei.  
		save_best_metric: Beste Metric wird als Textdatei gespeichert. Wird im Modell genutzt um Aktuelle Metric mit der besten Metric zu vergleichen.  
						  Ist die Aktuelle Metric präziser ersetzt sie die Beste Metrik und das Modell wird gespeichert.  
		save_best_metric_info: Beste Metric wird mit bester Epoche und Zeitstempel in einer Textdatei gespeichert.  
		Load_best_metric: Beste Medric wird geladen. Wenn durch eine File iteriert wird wird nicht durch jedes Symbol einzel sondern durch jede Zeile durchiteriert.  
		create_writer: Schreibt die Loss und Dicemetriken in Verzeichnis, lässt sich mit Tensorboard visualisieren.  
!!++	save_nifti: Nimmt eine Liste mit den predictions an (vorher hat sie nur einen Tensor angenommen). -> Prediction wird in einer Nifti-Datei gespeichert.   
		number_of_classes: Gibt die Anzahl an Klassen zurück  
		remove_directory_recursive: Löscht ein Verzeichnis mit all seinen Daten.  
		rescale_predictions: Rescaled die predictions auf die höhen, breiten und tiefen dimension der Originalbilder.  
		
---------predictions-----------  
	Es wird ein Monai transformer namen Activations importiert. Diese gibt die möglichkeit auf die Daten eine Aktivierungsfunktion wirken schon bevor sie ins Netz gehen.  
	Das hat den Vorteil das wie bei anderen Transformern die daten gleichgemacht werden, es ist auch möglich die Aktivierungsmuster des Modells zu visualisieren.  
	Ich habe die Metriken (Dice loss, dice metric) mit matplotlic visualisiert.  
!!	sliding_window_inference: Anstatt den Input direkt in das Modell zu geben um eine Prediction auf das ganze Bild in einem Durchgang zu bekommen nutze ich die sliding_window_inference methode. Diese iteriert in kleinen Schritten über das Bild. Anstatt in einem schritt an das Modell zu übergeben,  
							  wird das Bild in kleine überlappende Fenster zerteilt und das Modell wird für jedes dieser Fenster einzeln aufgerufen. Dies kann zu genaueren Ergebnissen führen.  
							  Möglicherweise benötige ich die methode nicht da mein Bild nicht so groß ist und momentan trotzdem das ganze Bild mit einmal reingesteckt wird, das macht die ganze Sache sinnlos.  
							  Ich könnte für eine höhere genauigkeit die Roi_size (größe der Patches) senken.  
							  Die sw_batch_size gibt wie viele patches auf einmal in das Netz gesteckt werden.   
!!	Die Sigmoid Aktivierungsfunktion nach dem Forwardpass wandelt den Output in werte zwischen 0 - 1 um. Vielleicht ist das nicht nötig und ich kann die Werte auch so schon nutzen. Durch Skalierung zwischen 0 und 1 Werten die Plots sehr Hell.   
	Beim Plotten ist es nicht erforderlich eine Tiefendimension hinzuzufügen da ein Grauwertbild keine Tiefe besitzt.  
	[0, 0, :, :, i]
	1. Batchdimension  
	2. Channeldimension (ist der einzige Channel deshalb 0)  
	3. Höhe  
	4. Breite  
	5. Index  

----------plot------------  
	Beeinhaltet alle Funktionen zum plotten von Daten
	Funktionen:
		generate_mesh: Generiert ein Mesh auf basis der reskalierten predictions.  
		plot_mesh: Vertices sind Koordinatenpunkte im Raum welche die Ecken der Dreiecke definieren. Ein Vertices besteht aus einer Anzahl Vortex (Einzahl von Vertices),  
				   jeder Vortex enthält die Koordinaten (x, y ,z) zu einem Punkt im Raum.  
				   Faces definiert die Verbindung zwischen Vertices um eine Fläche oder ein Ploygon zu bilden.  
		plot_confusion_matrix: Erstellt eine Confusion Matrix welche die Labels und die predictions gegenüberstellt.    
		plot_image_label_prediction: Plottet mit hilfe von Matplolib einen Ausgewählten Datensatz (Image, label, binary prediction, multi prediction)  
		plot_metric: erstellt ein matplotlib diagramm für die Metriken: train/test loss und train/test metrik  

	
---------Tensorboard Tutorial--------  
	Um tensorboard auszuführen muss dieser Kommandozeilen Befehl eingegeben werden: tensorboard --logdir ..\runs  
	Um Tensorboard im Browser darzustellen muss dies Lokaladresse eingegeben werden: localhost:6007  

------------------------------------------------------------  
Ordner Strukture: Die Images und die zugehörigen Labels müssen in gleich benannten Ordner abgespeichert werden (z.B heart_01)  
  

------------------------------------------------------------  
Annotationen: Um die lesbarkeit des Codes zu erhöhen habe ich alle Parameter, Rückgabewerte und Variabeln mit Datentyp Annotationen versehen. 