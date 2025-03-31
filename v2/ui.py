from imports import *
from utils import *
from VideoReader import *
from threads import Yolo11AnalysisThread, Florence2AnalysisThread, ClipAnalysisThread, Yolo8AnalysisThread

class VideoAnalysisUI(QWidget):
    def __init__(self, base_dir):
        super().__init__()
        self.BASE_DIR = base_dir
        self.videoPath = None
        self.analysisFolder = os.path.join(self.BASE_DIR, "Analysis")
        self.loading_timer = QTimer()
        self.loading_counter = 0
        self.isSeeking = False
        # Initialisation de l'interface utilisateur
        self.initUI()
        # Chargement initial
        self.load_video_hierarchy()
        display_logo(self.BASE_DIR, self.imageViewer, self.displayStack, self.fileNameLabel,
            update_window_button_visibility=lambda visible: update_window_button_visibility(
                self.closeButton, self.prevButton, self.nextButton, self.fileNameLabel, visible
            ),
            update_videoBar_button_visibility=lambda visible: update_videoBar_button_visibility(
                self.videoSlider, self.playPauseButton, self.videoTimeLabel, visible
            )
        )

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
        self.fileNav.itemClicked.connect(
            lambda item: display_file(
                item,
                self.analysisFolder,
                self.logViewer,
                self.displayStack,
                self.mediaPlayer,
                self.videoViewer,
                self.imageViewer,
                self.fileNameLabel,
                update_window_button_visibility=lambda visible: update_window_button_visibility(
                    self.closeButton, self.prevButton, self.nextButton, self.fileNameLabel, visible
                ),
                update_videoBar_button_visibility=lambda visible: update_videoBar_button_visibility(
                    self.videoSlider, self.playPauseButton, self.videoTimeLabel, visible
                )
            )
        )

        FilesButtonLayout = QHBoxLayout()
        # Boutons "Exporter"
        self.exportButton = QPushButton("Exporter")
        self.exportButton.clicked.connect(lambda: export_item(self.fileNav, "Analysis"))
        FilesButtonLayout.addWidget(self.exportButton)
        # Boutons "Ajouter"
        self.AddButton = QPushButton("Ajouter")
        self.AddButton.clicked.connect(lambda: add_item(self.fileNav, self.analysisFolder))
        FilesButtonLayout.addWidget(self.AddButton)
        # Boutons "Supprimer"
        self.deleteButton = QPushButton("Supprimer")
        self.deleteButton.clicked.connect(lambda: delete_selected_item(self.fileNav, "Analysis"))
        FilesButtonLayout.addWidget(self.deleteButton)
        navLayout.addLayout(FilesButtonLayout)

        # Ajouter un espace flexible pour pousser les boutons vers le bas
        navLayout.addStretch()

        # Boutons "Sélectionner une vidéo"
        self.uploadButton = QPushButton("Sélectionner une vidéo")
        self.uploadButton.clicked.connect(self.upload_video)
        navLayout.addWidget(self.uploadButton)

        ModelFilesLayout = QHBoxLayout()
        # Liste déroulante pour sélectionner le modèle
        self.modelSelector = QComboBox()
        self.modelSelector.addItems(["Yolo11", "Yolo8", "Clip", "Florence-2"])  # Ajouter les modèles disponibles
        self.modelSelector.setCurrentText("Yolo11")  # Définir le modèle par défaut
        ModelFilesLayout.addWidget(self.modelSelector)
        # Zone de texte pour les fps
        self.fpsBox = QTextEdit()
        self.fpsBox.setPlaceholderText("15")
        self.fpsBox.setFixedHeight(30)
        self.fpsBox.setFixedWidth(40)
        ModelFilesLayout.addWidget(self.fpsBox)
        # Texte affichant les fps
        self.FPSLabel = QLabel("fps")
        self.FPSLabel.setWordWrap(True)
        ModelFilesLayout.addWidget(self.FPSLabel)
        # Texte affichant l'estimation
        self.TimeLabel = QLabel("~ 00:00:00")
        self.TimeLabel.setWordWrap(True)
        ModelFilesLayout.addWidget(self.TimeLabel)
        navLayout.addLayout(ModelFilesLayout)
        ## fonction get_fps + calculate_time_estimation + display_estimation

        # Chatbox
        chatLayout = QVBoxLayout()
        self.chatBox = QTextEdit()
        self.chatBox.setPlaceholderText("Entrez votre prompt ici...")
        self.chatBox.setFixedHeight(50)
        chatLayout.addWidget(self.chatBox)
        navLayout.addLayout(chatLayout)

        # Bouton d'analyse
        self.analyzeButton = QPushButton("Lancer l'analyse")
        self.analyzeButton.setEnabled(False)
        self.analyzeButton.clicked.connect(self.start_analysis)
        navLayout.addWidget(self.analyzeButton)

        # Texte affichant la vidéo sélectionnée ou "Aucune vidéo en attente"
        self.selectedVideoLabel = QLabel("Aucune vidéo en attente")
        self.selectedVideoLabel.setWordWrap(True)
        navLayout.addWidget(self.selectedVideoLabel)
        # Ajouter le widget de navigation à gauche
        self.splitter.addWidget(navWidget)


        # ============================
        # Zone de contenu (droite)
        # Créer un layout horizontal pour diviser en deux parties
        rightMainLayout = QHBoxLayout()
        ## ------ ##
        ## Partie gauche ## Contient le bouton toggleNavButton
        leftToggleLayout = QVBoxLayout()
        self.toggleNavButton = QPushButton(" ")  # Bouton pour basculer la navigation
        self.toggleNavButton.setFixedWidth(15)  # Largeur fixe pour le bouton
        self.toggleNavButton.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)  # Étendre verticalement
        self.toggleNavButton.setText("<")  # Texte initial du bouton (l'onglet de gauche est visible par défaut)
        self.toggleNavButton.clicked.connect(lambda: toggle_navigation(self.splitter, self.toggleNavButton, self.adjust_splitter_proportions))
        leftToggleLayout.addStretch()
        leftToggleLayout.addStretch()
        leftToggleLayout.addStretch()
        leftToggleLayout.addWidget(self.toggleNavButton)
        leftToggleLayout.addStretch()
        leftToggleLayout.addStretch()
        leftToggleLayout.addStretch()
        ## ------ ##
        ## Partie droite ## Contient le reste du layout
        rightContentLayout = QVBoxLayout()
        # Navbar pour le nom du fichier et les boutons ##
        titleBarLayout = QHBoxLayout()
        self.fileNameLabel = QLabel("")  # Par défaut, aucun fichier n'est affiché
        self.fileNameLabel.setStyleSheet("font-weight: bold;")  # Style pour le nom du fichier
        # Bouton "<" pour fichier précédent
        self.prevButton = QPushButton("<")
        self.prevButton.setFixedSize(30, 30)
        self.prevButton.setStyleSheet("QPushButton { font-weight: bold; }")
        self.prevButton.clicked.connect(
            lambda: display_previous_file(self.fileNav, lambda item: display_file(
                item,
                self.analysisFolder,
                self.logViewer,
                self.displayStack,
                self.mediaPlayer,
                self.videoViewer,
                self.imageViewer,
                self.fileNameLabel,
                update_window_button_visibility=lambda visible: update_window_button_visibility(
                    self.closeButton, self.prevButton, self.nextButton, self.fileNameLabel, visible
                ),
                update_videoBar_button_visibility=lambda visible: update_videoBar_button_visibility(
                    self.videoSlider, self.playPauseButton, self.videoTimeLabel, visible
                )
            ))
        )
        # Bouton ">" pour fichier suivant
        self.nextButton = QPushButton(">")
        self.nextButton.setFixedSize(30, 30)
        self.nextButton.setStyleSheet("QPushButton { font-weight: bold; }")
        self.nextButton.clicked.connect(
            lambda: display_next_file(self.fileNav, lambda item: display_file(
                item,
                self.analysisFolder,
                self.logViewer,
                self.displayStack,
                self.mediaPlayer,
                self.videoViewer,
                self.imageViewer,
                self.fileNameLabel,
                update_window_button_visibility=lambda visible: update_window_button_visibility(
                    self.closeButton, self.prevButton, self.nextButton, self.fileNameLabel, visible
                ),
                update_videoBar_button_visibility=lambda visible: update_videoBar_button_visibility(
                    self.videoSlider, self.playPauseButton, self.videoTimeLabel, visible
                )
            ))
        )
        # Bouton "X" pour fermer
        self.closeButton = QPushButton("X")
        self.closeButton.setFixedSize(30, 30)
        self.closeButton.setStyleSheet("QPushButton { font-weight: bold; color: red; }")
        self.closeButton.clicked.connect(
            lambda: display_logo(
                self.BASE_DIR,
                self.imageViewer,
                self.displayStack,
                self.fileNameLabel,
                update_window_button_visibility=lambda visible: update_window_button_visibility(
                    self.closeButton, self.prevButton, self.nextButton, self.fileNameLabel, visible
                ),
                update_videoBar_button_visibility=lambda visible: update_videoBar_button_visibility(
                    self.videoSlider, self.playPauseButton, self.videoTimeLabel, visible
                )
            )
        )
        # Ajouter les widgets au layout
        titleBarLayout.addWidget(self.fileNameLabel)
        titleBarLayout.addStretch()  # Pousse les boutons à droite
        titleBarLayout.addWidget(self.prevButton)
        titleBarLayout.addWidget(self.nextButton)
        titleBarLayout.addWidget(self.closeButton)
        # Ajouter la barre au layout de droite
        rightContentLayout.addLayout(titleBarLayout)

        ## Différentes fenêtres pour afficher les fichiers ##
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
        rightContentLayout.addWidget(self.displayStack)

        ## Barre pour les contrôles vidéo (videoBar) ##
        self.videoBarLayout = QHBoxLayout()
        self.playPauseButton = QPushButton("⏸")  # Bouton Play/Pause
        self.playPauseButton.setFixedSize(30, 30)
        self.playPauseButton.clicked.connect(self.toggle_video_playback)
        self.videoBarLayout.addWidget(self.playPauseButton)
        self.videoSlider = QSlider(Qt.Orientation.Horizontal)  # Slider pour la progression
        self.videoSlider.setRange(0, 100)  # Valeurs par défaut, ajustées dynamiquement
        self.videoSlider.sliderPressed.connect(lambda: on_slider_pressed(self))
        self.videoSlider.sliderReleased.connect(lambda: on_slider_released(self))
        self.videoBarLayout.addWidget(self.videoSlider)
        self.videoTimeLabel = QLabel("00:00 / 00:00")  # Texte pour le temps de la vidéo
        self.videoTimeLabel.setFixedWidth(100)
        self.videoBarLayout.addWidget(self.videoTimeLabel)
        # Ajouter la videoBar au layout de droite
        rightContentLayout.addLayout(self.videoBarLayout)
        # Masquer la videoBar par défaut
        self.videoBarLayout.setEnabled(False)
        self.videoBarLayout.setContentsMargins(0, 0, 0, 0)

        # Ajouter les deux parties au layout principal
        rightMainLayout.addLayout(leftToggleLayout)  # Partie gauche
        rightMainLayout.addLayout(rightContentLayout)  # Partie droite
        # Ajouter le layout principal au splitter
        rightWidget = QWidget()
        rightWidget.setLayout(rightMainLayout)
        self.splitter.addWidget(rightWidget)

        # ============================
        # Configuration finale
        # ============================
        # Ajouter le splitter au layout principal
        mainLayout.addWidget(self.splitter)
        self.setLayout(mainLayout)
        # Initialisation des proportions du splitter
        self.adjust_splitter_proportions(hidden=False)  # Par défaut, le panneau de gauche est visible
        # Initialisation du lecteur multimédia
        self.mediaPlayer = QMediaPlayer()
        self.mediaPlayer.positionChanged.connect(self.update_video_progress)
        self.mediaPlayer.setVideoOutput(self.videoViewer)
        # Masquer le bouton "X" au démarrage
        update_window_button_visibility(self.closeButton, self.prevButton, self.nextButton, self.fileNameLabel, False)


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
            on_video_loaded(self)

            # Extraire le nom de la vidéo et le modèle sélectionné
            video_name = os.path.splitext(os.path.basename(video_path))[0]  # Nom sans extension
            model_name = self.modelSelector.currentText()  # Récupérer le modèle sélectionné
            video_duration_seconds = self.mediaPlayer.duration() // 1000  # Durée de la vidéo en secondes
    
            fps = self.get_target_fps() # Récupérer le fps depuis fpsBox
            # Calculer l'estimation du temps de traitement
            estimated_time = calculate_time_estimation(video_duration_seconds, fps, model_name)
            self.TimeLabel.setText(f"~ {estimated_time}") # Mettre à jour TimeLabel avec l'estimation

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
        target_fps = self.get_target_fps()  # Récupérer la valeur de fpsBox

        # Configurer le timer pour mettre à jour le texte toutes les 500 ms
        self.loading_timer.timeout.connect(lambda: self.update_loading_text(video_name, model_name))
        self.loading_timer.start(500)  # Mettre à jour toutes les 500 ms

        print(f"🔄 Analyse de {video_name}.mp4 par {model_name} en cours avec {target_fps} fps...")

        if model_name == "Clip":
            self.analysisThread = ClipAnalysisThread(self.analysisFolder, self.videoPath, target_fps)
        elif model_name == "Florence-2":
            self.analysisThread = Florence2AnalysisThread(self.analysisFolder, self.videoPath, target_fps)
        elif model_name == "Yolo11":
            self.analysisThread = Yolo11AnalysisThread(self.videoPath, model_name, target_folder, target_fps=target_fps)
        elif model_name == "Yolo8":
            self.analysisThread = Yolo8AnalysisThread(self.analysisFolder, self.videoPath, target_fps)
        else:
            print(f"❌ Modèle non pris en charge : {model_name}")
            self.loading_timer.stop()
            self.loading_counter = 0
            return

        # Connecter le signal de fin d'analyse
        self.analysisThread.analysis_finished.connect(self.on_analysis_finished)
        self.analysisThread.start()
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
        ] # Alterner entre les différents états
        self.selectedVideoLabel.setText(loading_states[self.loading_counter])
        self.loading_counter = (self.loading_counter + 1) % len(loading_states)

    # ============================
    # Gestion des fichiers
    # ============================
    def load_video_hierarchy(self):
        """Charge la hiérarchie des fichiers dans le QTreeWidget."""
        load_video_hierarchy(self.fileNav, "Analysis")
    def toggle_video_playback(self):
        """Met en pause ou reprend la lecture de la vidéo."""
        toggle_video_playback(self.mediaPlayer, self.playPauseButton)
    def update_video_progress(self):
        """Met à jour le slider et le texte du temps en fonction de la progression de la vidéo."""
        update_video_progress(self.mediaPlayer, self.videoSlider, self.videoTimeLabel, self.isSeeking, position=self.mediaPlayer.position())
    def seek_video(self, position):
        """Permet de naviguer dans la vidéo en cliquant sur le slider."""
        seek_video(self.mediaPlayer, position, self.videoSlider)
    def adjust_splitter_proportions(self, hidden):
        """Ajuste les proportions des widgets dans le splitter."""
        if hidden:
            # Si le panneau de gauche est masqué, donner tout l'espace au panneau de droite
            self.splitter.setSizes([0, self.toggleNavButton.width(), self.width() - self.toggleNavButton.width()])
        else:
            # Si le panneau de gauche est visible, réduire davantage la partie gauche et centrale
            left_width = int(self.width() * 0.3)  # 30% pour la partie gauche
            right_width = self.width() - left_width  # Le reste pour la partie droite
            self.splitter.setSizes([left_width, right_width])
    def get_target_fps(self):
        """Récupère la valeur de fpsBox et retourne une valeur bornée entre 1 et 60."""
        try:
            fps = int(self.fpsBox.toPlainText().strip())
            if fps < 1:
                return 1  # Valeur minimale
            elif fps > 60:
                return 60  # Valeur maximale
            return fps
        except ValueError:
            return 15  # Valeur par défaut si la saisie est invalide