import os
import shutil
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QPushButton, QStackedWidget, QSplitter, QTextEdit, QLabel, QComboBox, QHBoxLayout
)
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QUrl, Qt, QTimer
from PyQt6.QtGui import QPixmap, QIcon
from utils import upload_video, load_video_hierarchy, delete_selected_item, export_item
from threads import Yolo11AnalysisThread, Florence2AnalysisThread, ClipAnalysisThread, Yolo8AnalysisThread

class VideoAnalysisUI(QWidget):
    def __init__(self, base_dir):
        super().__init__()
        self.BASE_DIR = base_dir
        self.videoPath = None
        self.analysisFolder = os.path.join(self.BASE_DIR, "Analysis")
        self.loading_timer = QTimer()
        self.loading_counter = 0
        # Initialisation de l'interface utilisateur
        self.initUI()
        # Chargement initial
        self.load_video_hierarchy()
        self.display_logo()

    # ============================
    # Initialisation de l'interface utilisateur
    # ============================

    def initUI(self):
        """Initialise l'interface utilisateur."""
        # Configuration de la fenêtre principale
        self.setWindowTitle("Video AI")
        self.setGeometry(100, 100, 900, 600)
        self.setWindowIcon(QIcon("logo.ico"))
        self.setFixedSize(900, 600)
        # Layout principal
        mainLayout = QVBoxLayout()
        self.splitter = QSplitter()

        # ============================
        # Barre de navigation (gauche)

        navLayout = QVBoxLayout()
        navWidget = QWidget()
        navWidget.setLayout(navLayout)
        # Arborescence des fichiers
        self.fileNav = QTreeWidget()
        self.fileNav.setHeaderLabel("Structure des fichiers")
        navLayout.addWidget(self.fileNav)
        # Connecter le signal après avoir défini self.fileNav
        self.fileNav.itemClicked.connect(self.display_file)
        # Liste déroulante pour sélectionner le modèle
        self.modelSelector = QComboBox()
        self.modelSelector.addItems(["Yolo11", "Yolo8", "Clip", "Florence-2"])  # Ajouter les modèles disponibles
        self.modelSelector.setCurrentText("Yolo11")  # Définir le modèle par défaut
        navLayout.addWidget(self.modelSelector)
        # Boutons de navigation
        self.uploadButton = QPushButton("Sélectionner une vidéo")
        self.uploadButton.clicked.connect(self.upload_video)
        navLayout.addWidget(self.uploadButton)
        self.analyzeButton = QPushButton("Lancer l'analyse")
        self.analyzeButton.setEnabled(False)
        self.analyzeButton.clicked.connect(self.start_analysis)
        navLayout.addWidget(self.analyzeButton)
        # Boutons "Exporter"
        buttonLayout = QHBoxLayout()
        self.exportButton = QPushButton("Exporter")
        self.exportButton.clicked.connect(lambda: export_item(self.fileNav, "Analysis"))
        buttonLayout.addWidget(self.exportButton)
        # Boutons "Supprimer"
        self.deleteButton = QPushButton("Supprimer")
        self.deleteButton.clicked.connect(lambda: delete_selected_item(self.fileNav, "Analysis"))
        buttonLayout.addWidget(self.deleteButton)
        navLayout.addLayout(buttonLayout)
        # Texte affichant la vidéo sélectionnée ou "Aucune vidéo en attente"
        self.selectedVideoLabel = QLabel("Aucune vidéo en attente")
        self.selectedVideoLabel.setWordWrap(True)
        navLayout.addWidget(self.selectedVideoLabel)
        # Ajouter un espace flexible pour pousser les boutons vers le bas
        navLayout.addStretch()
        # Chatbox avec un bouton "Envoyer"
        chatLayout = QVBoxLayout()
        self.chatBox = QTextEdit()
        self.chatBox.setPlaceholderText("Entrez votre message ici...")
        chatLayout.addWidget(self.chatBox)
        self.sendButton = QPushButton("Envoyer")
        self.sendButton.clicked.connect(self.send_message)
        chatLayout.addWidget(self.sendButton)
        navLayout.addLayout(chatLayout)
        # Ajouter le widget de navigation à gauche
        self.splitter.addWidget(navWidget)


        # ============================
        # Zone de contenu (droite)

        rightLayout = QVBoxLayout()

        # Barre contenant le nom du fichier et les boutons "<", ">", et "X"
        titleBarLayout = QHBoxLayout()
        self.fileNameLabel = QLabel("")  # Par défaut, aucun fichier n'est affiché
        self.fileNameLabel.setStyleSheet("font-weight: bold;")  # Style pour le nom du fichier

        # Bouton "<" pour fichier précédent
        self.prevButton = QPushButton("<")
        self.prevButton.setFixedSize(30, 30)
        self.prevButton.setStyleSheet("QPushButton { font-weight: bold; }")
        self.prevButton.clicked.connect(self.display_previous_file)

        # Bouton ">" pour fichier suivant
        self.nextButton = QPushButton(">")
        self.nextButton.setFixedSize(30, 30)
        self.nextButton.setStyleSheet("QPushButton { font-weight: bold; }")
        self.nextButton.clicked.connect(self.display_next_file)

        # Bouton "X" pour fermer
        self.closeButton = QPushButton("X")
        self.closeButton.setFixedSize(30, 30)
        self.closeButton.setStyleSheet("QPushButton { font-weight: bold; color: red; }")
        self.closeButton.clicked.connect(self.display_logo)

        # Ajouter les widgets au layout
        titleBarLayout.addWidget(self.fileNameLabel)
        titleBarLayout.addStretch()  # Pousse les boutons à droite
        titleBarLayout.addWidget(self.prevButton)
        titleBarLayout.addWidget(self.nextButton)
        titleBarLayout.addWidget(self.closeButton)

        # Ajouter la barre au layout de droite
        rightLayout.addLayout(titleBarLayout)

        # QStackedWidget pour afficher différents contenus
        self.displayStack = QStackedWidget()

        # Onglet pour afficher les logs
        self.logViewer = QTextEdit()
        self.logViewer.setReadOnly(True)
        self.displayStack.addWidget(self.logViewer)

        # Onglet pour la vidéo
        self.videoViewer = QVideoWidget()
        self.displayStack.addWidget(self.videoViewer)

        # Onglet pour afficher les fichiers JSON ou texte
        self.jsonViewer = QTextEdit()
        self.jsonViewer.setReadOnly(True)
        self.displayStack.addWidget(self.jsonViewer)

        # Onglet pour afficher les images
        self.imageViewer = QLabel()
        self.imageViewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.displayStack.addWidget(self.imageViewer)

        # Ajouter le QStackedWidget au layout de droite
        rightLayout.addWidget(self.displayStack)

        # Ajouter le layout de droite au splitter
        rightWidget = QWidget()
        rightWidget.setLayout(rightLayout)
        self.splitter.addWidget(rightWidget)

        # ============================
        # Configuration finale
        # ============================
        # Définir les proportions des widgets dans le splitter (1/3 à gauche, 2/3 à droite)
        self.splitter.setStretchFactor(0, 1)  # Barre de navigation (1 part)
        self.splitter.setStretchFactor(1, 2)  # Zone de contenu (2 parts)

        # Ajouter le splitter au layout principal
        mainLayout.addWidget(self.splitter)
        self.setLayout(mainLayout)

        # Initialisation des variables
        self.mediaPlayer = QMediaPlayer()
        self.mediaPlayer.setVideoOutput(self.videoViewer)

        # Masquer le bouton "X" au démarrage
        self.update_close_button_visibility(False)


    # ============================
    # Gestion des vidéos
    # ============================

    def upload_video(self):
        """Permet à l'utilisateur de sélectionner une vidéo et de la charger."""
        video_path = upload_video()
        if video_path:
            self.videoPath = video_path
            self.mediaPlayer.setSource(QUrl.fromLocalFile(video_path))
            self.mediaPlayer.play()
            self.analyzeButton.setEnabled(True)  # Activer le bouton d'analyse

            # Extraire le nom de la vidéo et le modèle sélectionné
            video_name = os.path.splitext(os.path.basename(video_path))[0]  # Nom sans extension
            model_name = self.modelSelector.currentText()  # Récupérer le modèle sélectionné
            # Générer un dossier unique sous Analysis
            base_folder = os.path.join(self.analysisFolder, f"{video_name}_{model_name}")
            target_folder = base_folder
            counter = 1
            while os.path.exists(target_folder):  # Si le dossier existe déjà, incrémenter le suffixe
                target_folder = f"{base_folder}({counter})"
                counter += 1

            os.makedirs(target_folder, exist_ok=True)  # Créer le dossier unique
            # Copier la vidéo dans le dossier
            target_path = os.path.join(target_folder, f"{video_name}.mp4")
            shutil.copy(video_path, target_path)
            # Mettre à jour le label pour afficher la vidéo sélectionnée
            self.selectedVideoLabel.setText(f"Vidéo sélectionnée : {video_name}.mp4")
            print(f"✅ Vidéo copiée dans : {target_path}")
            # Stocker le chemin du dossier pour l'analyse
            self.currentAnalysisFolder = target_folder

    def start_analysis(self):
        """Lance l'analyse de la vidéo sélectionnée dans un thread séparé."""
        if not self.videoPath:
            return

        video_name = os.path.splitext(os.path.basename(self.videoPath))[0]
        model_name = self.modelSelector.currentText()  # Récupérer le modèle sélectionné
        target_folder = self.currentAnalysisFolder  # Utiliser le dossier créé dans upload_video

        # Configurer le timer pour mettre à jour le texte toutes les 500 ms
        self.loading_timer.timeout.connect(lambda: self.update_loading_text(video_name, model_name))
        self.loading_timer.start(500)  # Mettre à jour toutes les 500 ms
        print(f"🔄 Analyse de {video_name}.mp4 par {model_name} en cours...")

        if model_name == "Clip":
            self.analysisThread = ClipAnalysisThread(self.analysisFolder, self.videoPath, target_fps=15)
            self.analysisThread.analysis_finished.connect(self.on_analysis_finished)
            self.analysisThread.start()

        elif model_name == "Florence-2":
            self.analysisThread = Florence2AnalysisThread(self.analysisFolder, self.videoPath, target_fps=15)
            self.analysisThread.analysis_finished.connect(self.on_analysis_finished)
            self.analysisThread.start()

        elif model_name == "Yolo11":
            self.analysisThread = Yolo11AnalysisThread(self.videoPath, model_name, self.analysisFolder)
            self.analysisThread.analysis_finished.connect(self.on_analysis_finished)
            self.analysisThread.start()

        elif model_name == "Yolo8":
            self.analysisThread = Yolo8AnalysisThread(self.analysisFolder, self.videoPath, target_fps=15)
            self.analysisThread.analysis_finished.connect(self.on_analysis_finished)
            self.analysisThread.start()
        else:
            print(f"❌ Modèle non pris en charge : {model_name}")
            self.loading_timer.stop()
            self.loading_counter = 0
            
        print("🔄 Analyse en cours...")

    def on_analysis_finished(self, analysed_video_path):
        """Gère la fin de l'analyse et affiche la vidéo analysée."""
        self.loading_timer.stop()
        self.loading_counter = 0
        print(f"✅ Analyse terminée. Vidéo analysée : {analysed_video_path}")
        # Charger et lire la vidéo analysée
        if os.path.exists(analysed_video_path):
            self.mediaPlayer.setSource(QUrl.fromLocalFile(analysed_video_path))
            self.mediaPlayer.play()
            self.displayStack.setCurrentWidget(self.videoViewer)
            print(f"🎥 Vidéo analysée affichée : {analysed_video_path}")
        # Réinitialiser le label
        self.selectedVideoLabel.setText("Aucune vidéo en attente")
        self.videoPath = None
        # Recharger la hiérarchie des fichiers
        self.load_video_hierarchy()

    def update_loading_text(self, video_name, model_name):
        """Met à jour le texte du label pour simuler un chargement."""
        loading_states = [
            f"Analyse de {video_name}.mp4 par {model_name} en cours",
            f"Analyse de {video_name}.mp4 par {model_name} en cours.",
            f"Analyse de {video_name}.mp4 par {model_name} en cours..",
            f"Analyse de {video_name}.mp4 par {model_name} en cours...",
            f"Analyse de {video_name}.mp4 par {model_name} en cours...",
            f"Analyse de {video_name}.mp4 par {model_name} en cours..."
        ]
        # Alterner entre les différents états
        self.selectedVideoLabel.setText(loading_states[self.loading_counter])
        self.loading_counter = (self.loading_counter + 1) % len(loading_states)


    # ============================
    # Gestion des fichiers
    # ============================

    def load_video_hierarchy(self):
        """Charge la hiérarchie des fichiers dans le QTreeWidget."""
        load_video_hierarchy(self.fileNav, "Analysis")

    def display_previous_file(self):
        """Affiche le fichier précédent dans la hiérarchie."""
        current_item = self.fileNav.currentItem()
        if current_item:
            prev_item = self.fileNav.itemAbove(current_item)
            while prev_item:
                # Si l'élément précédent est un dossier, parcourir ses fichiers depuis le dernier
                if prev_item.childCount() > 0:  # Vérifie si c'est un dossier
                    last_child = prev_item.child(prev_item.childCount() - 1)
                    self.fileNav.setCurrentItem(last_child)
                    self.display_file(last_child)
                    return
                elif prev_item.childCount() == 0:  # Si c'est un fichier
                    self.fileNav.setCurrentItem(prev_item)
                    self.display_file(prev_item)
                    return
                prev_item = self.fileNav.itemAbove(prev_item)
            print("❌ Aucun fichier précédent.")  # Aucun fichier précédent trouvé

    def display_next_file(self):
        """Affiche le fichier suivant dans la hiérarchie."""
        current_item = self.fileNav.currentItem()
        if current_item:
            next_item = self.fileNav.itemBelow(current_item)
            while next_item:
                # Si l'élément suivant est un dossier, parcourir ses fichiers depuis le premier
                if next_item.childCount() > 0:  # Vérifie si c'est un dossier
                    first_child = next_item.child(0)
                    self.fileNav.setCurrentItem(first_child)
                    self.display_file(first_child)
                    return
                elif next_item.childCount() == 0:  # Si c'est un fichier
                    self.fileNav.setCurrentItem(next_item)
                    self.display_file(next_item)
                    return
                next_item = self.fileNav.itemBelow(next_item)
            print("❌ Aucun fichier suivant.")  # Aucun fichier suivant trouvé

    def on_file_selected(self):
        """Affiche le contenu du fichier sélectionné dans l'onglet de droite."""
        selected_item = self.fileNav.currentItem()
        if not selected_item:
            return

        file_path = os.path.join("Analysis", selected_item.text(0))  # Construire le chemin complet
        if os.path.isfile(file_path):
            if file_path.endswith(".json"):
                # Afficher le contenu JSON
                with open(file_path, "r") as f:
                    content = f.read()
                self.jsonViewer.setPlainText(content)
                self.displayStack.setCurrentWidget(self.jsonViewer)
            elif file_path.endswith(".png") or file_path.endswith(".jpg"):
                # Afficher l'image
                pixmap = QPixmap(file_path)
                self.imageViewer.setPixmap(pixmap)
                self.displayStack.setCurrentWidget(self.imageViewer)
            elif file_path.endswith(".mp4"):
                # Lire la vidéo
                self.mediaPlayer.setSource(QUrl.fromLocalFile(file_path))
                self.mediaPlayer.play()
                self.displayStack.setCurrentWidget(self.videoViewer)

    def update_close_button_visibility(self, visible):
        """Met à jour la visibilité du bouton 'X'."""
        self.closeButton.setVisible(visible)
        self.prevButton.setVisible(visible)
        self.nextButton.setVisible(visible)
        self.fileNameLabel.setVisible(visible)

    def display_logo(self):
        """Affiche le logo par défaut dans l'onglet de droite."""
        logo_path = os.path.join(self.BASE_DIR, "logo.png")
        print(f"🔍 Chemin du logo : {logo_path}")  # Log pour vérifier le chemin
        pixmap = QPixmap(logo_path)
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(
                150, 150,  # Taille du logo
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.imageViewer.setPixmap(scaled_pixmap)
            self.displayStack.setCurrentWidget(self.imageViewer)
            self.fileNameLabel.setText("")  # Réinitialiser le nom du fichier
            self.update_close_button_visibility(False)  # Masquer le bouton "X"
        else:
            print("❌ Erreur : Impossible de charger le logo par défaut.")

    def display_file(self, item):
        """Affiche le contenu du fichier sélectionné dans l'onglet de droite ou le logo par défaut si le fichier est illisible."""
        if item is None or not isinstance(item, QTreeWidgetItem):
            print("❌ Aucun élément valide sélectionné. Affichage du logo par défaut.")
            self.display_logo()
            return

        # Récupérer le nom du fichier sélectionné
        selected_file = item.text(0)
        print(f"🔍 Fichier sélectionné : {selected_file}")

        # Construire le chemin complet
        file_path = selected_file
        current_item = item
        while current_item.parent() is not None:
            current_item = current_item.parent()
            file_path = os.path.join(current_item.text(0), file_path)

        file_path = os.path.join(self.analysisFolder, file_path)

        # Vérifier si le fichier existe
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            print(f"❌ Fichier introuvable : {file_path}")
            self.display_logo()
            return

        # Afficher le nom du fichier et les boutons
        self.fileNameLabel.setText(selected_file)
        self.update_close_button_visibility(True)

        # Afficher le contenu en fonction du type de fichier
        try:
            if file_path.endswith(".json"):
                with open(file_path, "r") as f:
                    content = f.read()
                self.jsonViewer.setPlainText(content)
                self.displayStack.setCurrentWidget(self.jsonViewer)
            elif file_path.endswith(".png") or file_path.endswith(".jpg"):
                # Charger l'image et la redimensionner pour qu'elle s'adapte à la fenêtre
                pixmap = QPixmap(file_path)
                scaled_pixmap = pixmap.scaled(
                    self.imageViewer.size(),  # Taille du QLabel
                    Qt.AspectRatioMode.KeepAspectRatio,  # Conserver les proportions
                    Qt.TransformationMode.SmoothTransformation  # Transformation lissée
                )
                self.imageViewer.setPixmap(scaled_pixmap)
                self.displayStack.setCurrentWidget(self.imageViewer)
            elif file_path.endswith(".mp4"):
                self.mediaPlayer.setSource(QUrl.fromLocalFile(file_path))
                self.mediaPlayer.play()
                self.displayStack.setCurrentWidget(self.videoViewer)
            else:
                # Si le format est illisible, afficher un message dans le logViewer
                self.logViewer.setPlainText(f"❌ Format de fichier non pris en charge : {selected_file}")
                self.displayStack.setCurrentWidget(self.logViewer)
        except Exception as e:
            print(f"❌ Erreur lors de l'affichage du fichier : {e}")
            self.logViewer.setPlainText(f"❌ Erreur lors de l'affichage du fichier : {selected_file}")
            self.displayStack.setCurrentWidget(self.logViewer)


    # ============================
    # Autres utilitaires
    # ============================

    def send_message(self):
        """Méthode appelée lorsque le bouton 'Envoyer' est cliqué."""
        message = self.chatBox.toPlainText()
        if message.strip():
            print(f"Message envoyé : {message}")  # Remplacez par le traitement du message
            self.chatBox.clear()  # Effacer la chatbox après l'envoi
        else:
            print("Aucun message à envoyer.")
