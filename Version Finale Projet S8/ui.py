from imports import *
from utils import *
from VideoReader import *
from upload import *
from AI_Models.Clip_Analysis import *

class VideoAnalysisUI(QWidget):
    def __init__(self, base_dir, bank_dir, analysis_dir):
        super().__init__()
        self.BASE_DIR = base_dir
        self.bankFolder = bank_dir
        self.analysisFolder = analysis_dir
        self.videoPath = None
        self.loading_timer = QTimer()
        self.loading_counter = 0
        self.isSeeking = False
        # Initiate UI
        self.initUI()
        # Initial loading
        self.load_hierarchy(self.bankNav, self.bankFolder)
        self.load_hierarchy(self.fileNav, self.analysisFolder)
        display_readme(
            self.BASE_DIR,
            self.logViewer,
            self.displayStack,
            self.fileNameLabel,
            lambda visible: update_window_button_visibility(
                self.closeButton, self.prevButton, self.nextButton, self.fileNameLabel, self.deleteButton, self.exportButton, self.renameButton, visible
            ),
            lambda visible: update_videoBar_button_visibility(
                self.videoSlider, self.playPauseButton, self.videoTimeLabel, visible
            ),
            toggleReadmeButton=self.toggleReadmeButton
        )

    # ============================
    # UI initialisation
    # ============================
    def initUI(self):
        # Main screen configuration
        self.setWindowTitle("Gen.Surveillance")
        self.setGeometry(100, 100, 900, 600)
        self.setWindowIcon(QIcon("assets/logo.ico"))
        self.setFixedSize(900, 600)
        # Main layout
        mainLayout = QVBoxLayout()
        self.splitter = QSplitter()

        # ============================
        # Left navbar
        navLayout = QVBoxLayout()
        navWidget = QWidget()
        navWidget.setLayout(navLayout)
        
        # Create a container for the header and QTreeWidget
        bankContainer = QWidget()
        bankLayout = QVBoxLayout(bankContainer)
        bankLayout.setContentsMargins(0, 0, 0, 0)
        bankLayout.setSpacing(5)
        # Create a custom widget for the header
        headerWidget = QWidget()
        headerLayout = QHBoxLayout(headerWidget)
        headerLayout.setContentsMargins(0, 0, 0, 0) 
        headerLayout.setSpacing(5) 

        # Add a label for the title
        headerLabel = QLabel("Bank")
        headerLabel.setStyleSheet("font-weight: bold;")
        headerLayout.addWidget(headerLabel)
        # Add a label for the extraction message
        self.extractionLabel = QLabel("")
        self.extractionLabel.setWordWrap(True)
        self.extractionLabel.setVisible(False)  # Hide by default
        headerLayout.addWidget(self.extractionLabel)
        # Add a QLabel for the loading GIF
        self.extractionGifLabel = QLabel()
        self.extractionMovie = QMovie(os.path.join(self.BASE_DIR, "assets/loading.gif"))
        self.extractionGifLabel.setFixedSize(25, 25)
        self.extractionMovie.setScaledSize(self.extractionGifLabel.size())
        self.extractionGifLabel.setMovie(self.extractionMovie)
        self.extractionGifLabel.setVisible(False)  # Hide by default
        headerLayout.addWidget(self.extractionGifLabel)
        # Clean db button to display the database
        CleandbButton = QPushButton("🧹")
        CleandbButton.setFixedSize(25, 25)
        CleandbButton.setToolTip("Display database")
        CleandbButton.clicked.connect(lambda: faiss_client.reset_database(os.path.join(self.BASE_DIR, "data.faiss")))
        headerLayout.addWidget(CleandbButton)
        # Add a button to display the database
        dbButton = QPushButton("🏦")
        dbButton.setFixedSize(25, 25)
        dbButton.setToolTip("Display database")
        dbButton.clicked.connect(lambda: faiss_client.display_collections())
        headerLayout.addWidget(dbButton)
        # Adder a button in the header
        headerButton = QPushButton("➕")
        headerButton.setFixedSize(25, 25)
        headerButton.setToolTip("Add new item")
        headerButton.clicked.connect(
            lambda: add_item(
                self.bankFolder,
                faiss_client,
                self.extractionLabel,
                self.extractionGifLabel,
                self.extractionMovie,
                self.bankNav
            )
        )
        headerLayout.addWidget(headerButton)
        # Adder le widget d'en-tête au conteneur
        bankLayout.addWidget(headerWidget)

        # Add a button in the header
        self.bankNav = QTreeWidget()
        self.bankNav.setObjectName("bankNav")
        self.bankNav.setHeaderHidden(True)
        bankLayout.addWidget(self.bankNav)
        # Add the container to the main layout
        navLayout.addWidget(bankContainer)

        # Connect events of bankNav
        self.bankNav.focusInEvent = lambda event: self.set_active_tree_widget(self.bankNav)
        self.bankNav.itemClicked.connect(
            lambda item: display_file(
                item,
                self.bankFolder,
                self.logViewer,
                self.displayStack,
                self.mediaPlayer,
                self.videoViewer,
                self.imageViewer,
                self.fileNameLabel,
                update_window_button_visibility=lambda visible: update_window_button_visibility(
                    self.closeButton, self.prevButton, self.nextButton, self.fileNameLabel, self.deleteButton, self.exportButton, self.renameButton, visible
                ),
                update_videoBar_button_visibility=lambda visible: update_videoBar_button_visibility(
                    self.videoSlider, self.playPauseButton, self.videoTimeLabel, visible
                ),
                toggleReadmeButton=self.toggleReadmeButton 
            )
        )
 

        # Create a container for the header and QTreeWidget
        analysisContainer = QWidget()
        analysisLayout = QVBoxLayout(analysisContainer)
        analysisLayout.setContentsMargins(0, 0, 0, 0)  # delete margins
        analysisLayout.setSpacing(5) # spacing between elements
        # create a custom widget for the header
        analysisHeaderWidget = QWidget()
        analysisHeaderLayout = QHBoxLayout(analysisHeaderWidget)
        analysisHeaderLayout.setContentsMargins(0, 0, 0, 0)
        analysisHeaderLayout.setSpacing(5) 

        # Add a label for the title
        analysisHeaderLabel = QLabel("Analysis")
        analysisHeaderLabel.setStyleSheet("font-weight: bold;")
        analysisHeaderLayout.addWidget(analysisHeaderLabel)
        # Add a button in the header
        analysisLayout.addWidget(analysisHeaderWidget)

        # Add a button in the header
        self.fileNav = QTreeWidget()
        self.fileNav.setObjectName("fileNav")
        self.fileNav.setHeaderHidden(True)  # hide header by default
        analysisLayout.addWidget(self.fileNav)
        # Add the container to the main layout
        navLayout.addWidget(analysisContainer)

        # Connect events of fileNav
        self.fileNav.focusInEvent = lambda event: self.set_active_tree_widget(self.fileNav)
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
                    self.closeButton, self.prevButton, self.nextButton, self.fileNameLabel, self.deleteButton, self.exportButton, self.renameButton, visible
                ),
                update_videoBar_button_visibility=lambda visible: update_videoBar_button_visibility(
                    self.videoSlider, self.playPauseButton, self.videoTimeLabel, visible
                ),
                toggleReadmeButton=self.toggleReadmeButton 
            )
        )


        # stretch elements with empty space
        navLayout.addStretch()
        ModelFilesLayout = QHBoxLayout()
        # Text displaying fps
        self.ClipLabel = QLabel("Clip")
        self.ClipLabel.setWordWrap(True)
        ModelFilesLayout.addWidget(self.ClipLabel)
        # Chatbox
        chatLayout = QVBoxLayout()
        self.chatBox = QTextEdit()
        self.chatBox.setPlaceholderText("Enter your prompt here...")
        self.chatBox.setFixedHeight(50)
        chatLayout.addWidget(self.chatBox)
        navLayout.addLayout(chatLayout)
        # Analysis button
        self.analyzeButton = QPushButton("Start analysis")
        self.analyzeButton.clicked.connect(self.start_analysis)
        navLayout.addWidget(self.analyzeButton)

        # Info text and loading GIF
        loadingLayout = QHBoxLayout()
        self.selectedVideoLabel = QLabel("")
        self.selectedVideoLabel.setWordWrap(True)
        self.loadingGifLabel = QLabel()  # QLabel to display GIF
        self.loadingMovie = QMovie(os.path.join(BASE_DIR, "assets/loading.gif"))
        self.loadingGifLabel.setFixedSize(30, 30)
        self.loadingMovie.setScaledSize(self.loadingGifLabel.size()) 
        self.loadingGifLabel.setMovie(self.loadingMovie)
        self.loadingGifLabel.setVisible(False)  # hide by default

        loadingLayout.addWidget(self.selectedVideoLabel)
        loadingLayout.addWidget(self.loadingGifLabel) 
        navLayout.addLayout(loadingLayout)
        # Add to left navbar
        self.splitter.addWidget(navWidget)


        # ============================
        # Right layout 
        rightMainLayout = QHBoxLayout()
        ## ------ ##
        ## Left part ## toggleNavButton, hide/reveal left layout
        leftToggleLayout = QVBoxLayout()
        self.toggleNavButton = QPushButton(" ")
        self.toggleNavButton.setFixedWidth(15) 
        self.toggleNavButton.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)  # expand size verticaly
        self.toggleNavButton.setText("<")
        self.toggleNavButton.clicked.connect(lambda: toggle_navigation(self.splitter, self.toggleNavButton, self.adjust_splitter_proportions))
        leftToggleLayout.addStretch()
        leftToggleLayout.addStretch()
        leftToggleLayout.addStretch()
        leftToggleLayout.addWidget(self.toggleNavButton)
        leftToggleLayout.addStretch()
        leftToggleLayout.addStretch()
        leftToggleLayout.addStretch()
        ## ------ ##
        ## Right part ##
        rightContentLayout = QVBoxLayout()
        # top Navbar ## close, next/previous file, delete, export...
        titleBarLayout = QHBoxLayout()
        self.fileNameLabel = QLabel("")  # No name if no file selected
        self.fileNameLabel.setStyleSheet("font-weight: bold;")
        # "<" button for previous file
        self.prevButton = QPushButton("<")
        self.prevButton.setFixedSize(30, 30)
        self.prevButton.setStyleSheet("QPushButton { font-weight: bold; }")
        self.prevButton.clicked.connect(
            lambda: display_previous_file(
                self.get_active_tree_widget(),
                display_file,
                self.logViewer,
                self.displayStack,
                self.mediaPlayer,
                self.videoViewer,
                self.imageViewer,
                self.fileNameLabel,
                lambda visible: update_window_button_visibility(
                    self.closeButton, self.prevButton, self.nextButton, self.fileNameLabel, self.deleteButton, self.exportButton, self.renameButton, visible
                ),
                lambda visible: update_videoBar_button_visibility(
                    self.videoSlider, self.playPauseButton, self.videoTimeLabel, visible
                )
            )
        )
        # ">" for next file
        self.nextButton = QPushButton(">")
        self.nextButton.setFixedSize(30, 30)
        self.nextButton.setStyleSheet("QPushButton { font-weight: bold; }")
        self.nextButton.clicked.connect(
            lambda: display_next_file(
                self.get_active_tree_widget(),
                display_file,
                self.logViewer,
                self.displayStack,
                self.mediaPlayer,
                self.videoViewer,
                self.imageViewer,
                self.fileNameLabel,
                lambda visible: update_window_button_visibility(
                    self.closeButton, self.prevButton, self.nextButton, self.fileNameLabel, self.deleteButton, self.exportButton, self.renameButton, visible
                ),
                lambda visible: update_videoBar_button_visibility(
                    self.videoSlider, self.playPauseButton, self.videoTimeLabel, visible
                )
            )
        )
        # Add a button to toggle README
        self.toggleReadmeButton = QPushButton("❌")
        self.toggleReadmeButton.setFixedSize(30, 30)
        self.toggleReadmeButton.setToolTip("Toggle README")
        self.toggleReadmeButton.clicked.connect(self.toggle_readme)
        rightContentLayout.addWidget(self.toggleReadmeButton, alignment=Qt.AlignmentFlag.AlignRight)
        
        # "X" Button to close window
        self.closeButton = QPushButton("X")
        self.closeButton.setFixedSize(30, 30)
        self.closeButton.setStyleSheet("QPushButton { font-weight: bold; color: red; }")
        self.closeButton.clicked.connect(
            lambda: display_logo(
                self.imageViewer,
                self.displayStack,
                self.fileNameLabel,
                update_window_button_visibility=lambda visible: update_window_button_visibility(
                    self.closeButton, self.prevButton, self.nextButton, self.fileNameLabel, self.deleteButton, self.exportButton, self.renameButton, visible
                ),
                update_videoBar_button_visibility=lambda visible: update_videoBar_button_visibility(
                    self.videoSlider, self.playPauseButton, self.videoTimeLabel, visible
                ),
                toggleReadmeButton=self.toggleReadmeButton
            )
        )
        # "Rename" Button
        self.renameButton = QPushButton("✏️")
        self.renameButton.setFixedSize(25, 25)
        self.renameButton.setToolTip("Rename")
        self.renameButton.clicked.connect(
            lambda: rename_item(
                self.get_active_tree_widget(),
                self.BASE_DIR,
                self.mediaPlayer,
                self.fileNameLabel
            )
        )
        # "Export" Button
        self.exportButton = QPushButton("📤")
        self.exportButton.setFixedSize(25, 25)
        self.exportButton.setToolTip("Export")
        self.exportButton.clicked.connect(
            lambda: export_item(self.get_active_tree_widget(), self.BASE_DIR)
        )
        # "Delete" Button
        self.deleteButton = QPushButton("🗑️")
        self.deleteButton.setFixedSize(25, 25)
        self.deleteButton.setToolTip("Delete")
        self.deleteButton.clicked.connect(
            lambda: delete_item(
                self.get_active_tree_widget(),
                self.BASE_DIR,
                self.mediaPlayer,
                lambda: display_logo(
                    self.imageViewer,
                    self.displayStack,
                    self.fileNameLabel,
                    update_window_button_visibility=lambda visible: update_window_button_visibility(
                        self.closeButton, self.prevButton, self.nextButton, self.fileNameLabel, self.deleteButton, self.exportButton, self.renameButton, visible
                    ),
                    update_videoBar_button_visibility=lambda visible: update_videoBar_button_visibility(
                        self.videoSlider, self.playPauseButton, self.videoTimeLabel, visible
                    ),
                    toggleReadmeButton=self.toggleReadmeButton
                )
            )
        )

        # Add widgets to layout
        titleBarLayout.addWidget(self.fileNameLabel)
        titleBarLayout.addWidget(self.renameButton)
        titleBarLayout.addWidget(self.exportButton)
        titleBarLayout.addWidget(self.deleteButton)
        titleBarLayout.addStretch()  # Push buttons on right and left sides
        titleBarLayout.addWidget(self.prevButton)
        titleBarLayout.addWidget(self.nextButton)
        titleBarLayout.addWidget(self.closeButton)
        # Add navbar to right layout 
        rightContentLayout.addLayout(titleBarLayout)

        ## Select window elements by file type ##
        # QStackedWidget to display multiple content
        self.displayStack = QStackedWidget()
        # Text
        self.logViewer = QTextBrowser()
        self.logViewer.setOpenExternalLinks(True)
        self.displayStack.addWidget(self.logViewer)
        # Video
        self.videoViewer = QVideoWidget()
        self.displayStack.addWidget(self.videoViewer)
        # JSON
        self.jsonViewer = QTextEdit()
        self.jsonViewer.setReadOnly(True)
        self.displayStack.addWidget(self.jsonViewer)
        # Images
        self.imageViewer = QLabel()
        self.imageViewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.displayStack.addWidget(self.imageViewer)
        # Add QStackedWidget to the right layout
        rightContentLayout.addWidget(self.displayStack)

        ## Video controller ##
        self.videoBarLayout = QHBoxLayout()
        self.playPauseButton = QPushButton("⏸")  # Play/Pause
        self.playPauseButton.setFixedSize(30, 30)
        self.playPauseButton.clicked.connect(self.toggle_video_playback)
        self.videoBarLayout.addWidget(self.playPauseButton)
        self.videoSlider = QSlider(Qt.Orientation.Horizontal)  # Progression slider
        self.videoSlider.setRange(0, 100)  # Default values
        self.videoSlider.sliderPressed.connect(lambda: on_slider_pressed(self))
        self.videoSlider.sliderReleased.connect(lambda: on_slider_released(self))
        self.videoBarLayout.addWidget(self.videoSlider)
        self.videoTimeLabel = QLabel("00:00 / 00:00")  # Text for video duration
        self.videoTimeLabel.setFixedWidth(100)
        self.videoBarLayout.addWidget(self.videoTimeLabel)
        # Add video navbar to right layout
        rightContentLayout.addLayout(self.videoBarLayout)
        # Mask it by default
        self.videoBarLayout.setEnabled(False)
        self.videoBarLayout.setContentsMargins(0, 0, 0, 0)

        # Add layouts to the right panel
        rightMainLayout.addLayout(leftToggleLayout)
        rightMainLayout.addLayout(rightContentLayout)
        # Add main layout to splitter
        rightWidget = QWidget()
        rightWidget.setLayout(rightMainLayout)
        self.splitter.addWidget(rightWidget)

        # ============================
        # Final settings
        # ============================
        # Add splitter to main layout
        mainLayout.addWidget(self.splitter)
        self.setLayout(mainLayout)
        # Set up splitter proportions
        self.adjust_splitter_proportions(hidden=False)  # By default, left layout is not hidden
        # Set up multimediia player
        self.mediaPlayer = QMediaPlayer()
        self.mediaPlayer.positionChanged.connect(self.update_video_progress)
        self.mediaPlayer.setVideoOutput(self.videoViewer)
        # Mask close button visibility
        update_window_button_visibility(self.closeButton, self.prevButton, self.nextButton, self.fileNameLabel, self.deleteButton, self.exportButton, self.renameButton, False)


    # ============================
    # Video management
    # ============================

    def toggle_video_playback(self):
        """Pause/Play video"""
        toggle_video_playback(self.mediaPlayer, self.playPauseButton)
    def update_video_progress(self):
        """Update slider and time text with real video progression."""
        update_video_progress(self.mediaPlayer, self.videoSlider, self.videoTimeLabel, self.isSeeking, position=self.mediaPlayer.position())
    def seek_video(self, position):
        """Make navigation through video possible by clicking on the slider"""
        seek_video(self.mediaPlayer, position, self.videoSlider)

    def start_analysis(self):
        """
        Récupère le texte du chatBox et lance l'analyse avec le prompt.
        """
        prompt = self.chatBox.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Warning", "Please enter a prompt before starting the analysis.")
            return

        try:
            # Créer une instance de ClipAnalysis
            clip_analysis = ClipAnalysis(bank_path=self.bankFolder)
            # Appeler la méthode analyse_prompt sur l'instance
            result_folder = clip_analysis.analyse_prompt(self.analysisFolder, prompt)
            QMessageBox.information(self, "Analysis Complete", f"Results saved in: {result_folder}")
            self.load_hierarchy(self.fileNav, self.analysisFolder)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during analysis: {e}")
            print(f"❌ Error during analysis: {e}")

    def on_analysis_finished(self, analysed_video_path):
        """Handle analysis end and display appropriate file depending on the model"""
        self.loading_timer.stop()
        self.loading_counter = 0
        # Stop and hide GIF
        self.loadingMovie.stop()
        self.loadingGifLabel.setVisible(False)
        print(f"✅ Analyse terminée. Vidéo analysée : {analysed_video_path}")
        # Reinitiate process
        self.selectedVideoLabel.setText("")
        self.videoPath = None
        self.load_hierarchy()
        # Vérifier la base de données après l'analyse
        print("🔍 Vérification de la base de données après l'analyse...")



    # ============================
    # Files management
    # ============================

    def load_hierarchy(self, tree_widget, directory):
        """Load files tree in the specified QTreeWidget."""
        load_hierarchy(tree_widget, directory)
        
    def get_active_tree_widget(self):
        """Determine which tree widget is currently active."""
        if hasattr(self, "activeTreeWidget") and self.activeTreeWidget:
            return self.activeTreeWidget
        else:
            print("⚠️ No active tree widget detected. Defaulting to fileNav.")
            
            return self.fileNav
    def set_active_tree_widget(self, tree_widget):
        """Set the active tree widget."""
        self.activeTreeWidget = tree_widget
        print(f"🔍 Active tree widget set to: {tree_widget.objectName()}")

    def update_loading_text(self):
        # start/stop GIF animation
        if not self.loadingGifLabel.isVisible():
            self.loadingGifLabel.setVisible(True)
            self.loadingMovie.start()
        self.selectedVideoLabel.setText(f"Analyse en cours...")

    def adjust_splitter_proportions(self, hidden):
        """Adjust widgets proportion in the splitter"""
        if hidden:
            # if left pannel is hidden, right one take all screen width
            self.splitter.setSizes([0, self.toggleNavButton.width(), self.width() - self.toggleNavButton.width()])
        else:
            left_width = int(self.width() * 0.3)  # 30% for left part
            right_width = self.width() - left_width  # right take the rest
            self.splitter.setSizes([left_width, right_width])

    def get_target_fps(self):
        """Take fpsBox value and give back a value between 1 and 60."""
        try:
            fps = int(self.fpsBox.toPlainText().strip())
            if fps < 1:
                return 1  # Min value
            elif fps > 60:
                return 60  # Max value
            return fps
        except ValueError:
            return 2  # Default value if nothing token

    def toggle_readme(self):
        """Toggle between displaying README and the logo."""
        if self.toggleReadmeButton.text() == "ℹ️":
            # Afficher le README
            display_readme(
                self.BASE_DIR,
                self.logViewer,
                self.displayStack,
                self.fileNameLabel,
                lambda visible: update_window_button_visibility(
                    self.closeButton, self.prevButton, self.nextButton, self.fileNameLabel, self.deleteButton, self.exportButton, self.renameButton, visible
                ),
                lambda visible: update_videoBar_button_visibility(
                    self.videoSlider, self.playPauseButton, self.videoTimeLabel, visible
                ),
                toggleReadmeButton=self.toggleReadmeButton
            )
            self.toggleReadmeButton.setText("❌")  # Changer le texte du bouton
            self.toggleReadmeButton.setToolTip("Close README")
        else:
            # Afficher le logo
            display_logo(
                self.imageViewer,
                self.displayStack,
                self.fileNameLabel,
                lambda visible: update_window_button_visibility(
                    self.closeButton, self.prevButton, self.nextButton, self.fileNameLabel, self.deleteButton, self.exportButton, self.renameButton, visible
                ),
                lambda visible: update_videoBar_button_visibility(
                    self.videoSlider, self.playPauseButton, self.videoTimeLabel, visible
                ),
                toggleReadmeButton=self.toggleReadmeButton
            )
            self.toggleReadmeButton.setText("ℹ️")  # Changer le texte du bouton
            self.toggleReadmeButton.setToolTip("Show README")

    def closeEvent(self, event):
        """Handle application close event."""
        global upload_threads
        for thread in upload_threads:
            if thread.isRunning():
                print(f"🛑 Arrêt du thread : {thread}")
                thread.terminate()
                thread.wait()
        upload_threads.clear()
        event.accept()