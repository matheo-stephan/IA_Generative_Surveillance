from imports import *

def add_item(fileNav, base_dir):
    """Ouvre une fenêtre pour sélectionner un fichier et le copie dans le répertoire actuel."""
    # Ouvrir une boîte de dialogue pour sélectionner un fichier
    file_path, _ = QFileDialog.getOpenFileName(None, "Sélectionner un fichier", "", "Tous les fichiers (*)")
    
    if not file_path:  # Si aucun fichier n'est sélectionné
        return
    # Récupérer l'élément actuellement sélectionné dans l'arborescence
    current_item = fileNav.currentItem()
    if current_item:
        # Construire le chemin du répertoire actuel
        current_dir = os.path.join(base_dir, current_item.text(0))
        while current_item.parent() is not None:
            current_item = current_item.parent()
            current_dir = os.path.join(base_dir, current_item.text(0), current_dir)
    else:
        # Si aucun élément n'est sélectionné, utiliser le répertoire de base
        current_dir = base_dir
    if not os.path.exists(current_dir):
        QMessageBox.warning(None, "Erreur", f"Le répertoire sélectionné n'existe pas : {current_dir}")
        return
    # Copier le fichier dans le répertoire actuel
    try:
        shutil.copy(file_path, current_dir)
        QMessageBox.information(None, "Succès", f"Fichier ajouté avec succès dans : {current_dir}")
    except Exception as e:
        QMessageBox.critical(None, "Erreur", f"Impossible de copier le fichier : {e}")

def load_video_hierarchy(tree_widget, directory):
    """Charge la hiérarchie des fichiers dans un QTreeWidget."""
    tree_widget.clear()
    for item in sorted(os.listdir(directory)):
        item_path = os.path.join(directory, item)
        tree_item = QTreeWidgetItem(tree_widget, [item])
        if os.path.isdir(item_path):
            tree_item.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
            populate_tree(item_path, tree_item)

def populate_tree(directory, parent_item):
    """Remplit un QTreeWidget avec les fichiers et dossiers d'un répertoire."""
    for item in sorted(os.listdir(directory)):
        item_path = os.path.join(directory, item)
        tree_item = QTreeWidgetItem(parent_item, [item])
        if os.path.isdir(item_path):
            tree_item.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
            populate_tree(item_path, tree_item)

def delete_selected_item(tree_widget, base_dir):
    """Supprime l'élément sélectionné après confirmation."""
    selected_item = tree_widget.currentItem()
    if not selected_item:
        QMessageBox.warning(tree_widget, "Avertissement", "Aucun fichier ou dossier sélectionné.")
        return

    # Construire le chemin complet de l'élément sélectionné
    file_path = os.path.join(base_dir, selected_item.text(0))
    if not os.path.exists(file_path):
        QMessageBox.critical(tree_widget, "Erreur", f"Le fichier ou dossier '{file_path}' n'existe pas.")
        return

    # Demander confirmation avant de supprimer
    reply = QMessageBox.question(
        tree_widget,
        "Confirmation",
        f"Êtes-vous sûr de vouloir supprimer '{selected_item.text(0)}' ?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )

    if reply == QMessageBox.StandardButton.Yes:
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                shutil.rmtree(file_path)
            QMessageBox.information(tree_widget, "Succès", f"'{selected_item.text(0)}' a été supprimé.")
            # Supprimer l'élément de l'arborescence
            parent = selected_item.parent()
            if parent:
                parent.removeChild(selected_item)
            else:
                tree_widget.takeTopLevelItem(tree_widget.indexOfTopLevelItem(selected_item))
        except Exception as e:
            QMessageBox.critical(tree_widget, "Erreur", f"Erreur lors de la suppression : {e}")

def export_item(tree_widget, base_dir):
    """Exporte l'élément sélectionné dans le QTreeWidget sous forme de fichier ZIP."""
    selected_item = tree_widget.currentItem()
    if selected_item:
        item_path = os.path.join(base_dir, selected_item.text(0))
        zip_path, _ = QFileDialog.getSaveFileName(None, "Exporter sous forme de ZIP", "", "ZIP Files (*.zip)")
        if zip_path:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                if os.path.isdir(item_path):
                    for root, _, files in os.walk(item_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, item_path)
                            zipf.write(file_path, arcname)
                else:
                    zipf.write(item_path, os.path.basename(item_path))
            QMessageBox.information(None, "Exportation", f"Exportation réussie : {zip_path}")

def display_previous_file(fileNav, display_file):
    """Affiche le fichier précédent dans la hiérarchie."""
    current_item = fileNav.currentItem()
    if current_item:
        prev_item = fileNav.itemAbove(current_item)
        while prev_item:
            if prev_item.childCount() > 0:  # Si c'est un dossier, parcourir ses fichiers depuis le dernier
                last_child = prev_item.child(prev_item.childCount() - 1)
                fileNav.setCurrentItem(last_child)
                display_file(last_child)
                return
            elif prev_item.childCount() == 0:  # Si c'est un fichier
                fileNav.setCurrentItem(prev_item)
                display_file(prev_item)
                return
            prev_item = fileNav.itemAbove(prev_item)
    print("❌ Aucun fichier précédent.")  # Aucun fichier précédent trouvé

def display_next_file(fileNav, display_file):
    """Affiche le fichier suivant dans la hiérarchie."""
    current_item = fileNav.currentItem()
    if current_item:
        next_item = fileNav.itemBelow(current_item)
        while next_item:
            if next_item.childCount() > 0:  # Si c'est un dossier, parcourir ses fichiers depuis le premier
                first_child = next_item.child(0)
                fileNav.setCurrentItem(first_child)
                display_file(first_child)
                return
            elif next_item.childCount() == 0:  # Si c'est un fichier
                fileNav.setCurrentItem(next_item)
                display_file(next_item)
                return
            next_item = fileNav.itemBelow(next_item)
    print("❌ Aucun fichier suivant.")  # Aucun fichier suivant trouvé

def toggle_navigation(splitter, toggleNavButton, adjust_splitter_proportions):
    """Affiche ou masque le panneau de navigation et ajuste les proportions."""
    navWidget = splitter.widget(0)  # Récupérer le widget de navigation (gauche)
    if navWidget.isVisible():
        navWidget.hide()  # Masquer le panneau
        toggleNavButton.setText(">")  # Mettre le texte sur ">"
        adjust_splitter_proportions(hidden=True)
    else:
        navWidget.show()  # Afficher le panneau
        toggleNavButton.setText("<")  # Mettre le texte sur "<"
        adjust_splitter_proportions(hidden=False)

def on_file_selected(fileNav, analysisFolder, displayStack, mediaPlayer, videoViewer, jsonViewer, imageViewer):
    """Affiche le contenu du fichier sélectionné dans l'onglet de droite."""
    selected_item = fileNav.currentItem()
    if not selected_item:
        return
    file_path = os.path.join(analysisFolder, selected_item.text(0))  # Construire le chemin complet
    if os.path.isfile(file_path):
        if file_path.endswith(".json"):
            with open(file_path, "r") as f:
                content = f.read()
            jsonViewer.setPlainText(content)
            displayStack.setCurrentWidget(jsonViewer)
        elif file_path.endswith(".png") or file_path.endswith(".jpg"):
            pixmap = QPixmap(file_path)
            imageViewer.setPixmap(pixmap)
            displayStack.setCurrentWidget(imageViewer)
        elif file_path.endswith(".mp4"):
            mediaPlayer.setSource(QUrl.fromLocalFile(file_path))
            mediaPlayer.play()
            displayStack.setCurrentWidget(videoViewer)

def update_videoBar_button_visibility(videoSlider, playPauseButton, videoTimeLabel, visible):
    """Met à jour la visibilité de la barre vidéo."""
    videoSlider.setVisible(visible)
    playPauseButton.setVisible(visible)
    videoTimeLabel.setVisible(visible)

def update_window_button_visibility(closeButton, prevButton, nextButton, fileNameLabel, visible):
    """Met à jour la visibilité des boutons et du label."""
    closeButton.setVisible(visible)
    prevButton.setVisible(visible)
    nextButton.setVisible(visible)
    fileNameLabel.setVisible(visible)

def display_logo(BASE_DIR, imageViewer, displayStack, fileNameLabel, update_window_button_visibility, update_videoBar_button_visibility):
    """Affiche le logo par défaut dans l'onglet de droite."""
    logo_path = os.path.join(BASE_DIR, "logo.png")
    print(f"🔍 Chemin du logo : {logo_path}")  # Log pour vérifier le chemin
    pixmap = QPixmap(logo_path)
    if not pixmap.isNull():
        scaled_pixmap = pixmap.scaled(
            150, 150,  # Taille du logo
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        imageViewer.setPixmap(scaled_pixmap)
        displayStack.setCurrentWidget(imageViewer)
        fileNameLabel.setText("")  # Réinitialiser le nom du fichier
        update_window_button_visibility(False)  # Masquer les boutons
        update_videoBar_button_visibility(False)  # Masquer la barre vidéo
    else:
        print("❌ Erreur : Impossible de charger le logo par défaut.")

def display_file(item, analysisFolder, logViewer, displayStack, mediaPlayer, videoViewer, imageViewer, fileNameLabel, update_window_button_visibility, update_videoBar_button_visibility):
    """Affiche le contenu du fichier sélectionné dans l'onglet de droite ou le logo par défaut si le fichier est illisible."""
    if item is None or not isinstance(item, QTreeWidgetItem):
        print("❌ Aucun élément valide sélectionné. Affichage du logo par défaut.")
        display_logo(
            BASE_DIR=analysisFolder,  # Passez le répertoire de base
            imageViewer=imageViewer,
            displayStack=displayStack,
            fileNameLabel=fileNameLabel,
            update_window_button_visibility=update_window_button_visibility,
            update_videoBar_button_visibility=update_videoBar_button_visibility
        )()
        return

    selected_file = item.text(0)
    print(f"🔍 Fichier sélectionné : {selected_file}")

    file_path = selected_file
    current_item = item
    while current_item.parent() is not None:
        current_item = current_item.parent()
        file_path = os.path.join(current_item.text(0), file_path)
    file_path = os.path.join(analysisFolder, file_path)

    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        print(f"❌ Fichier introuvable : {file_path}")
        display_logo(
            BASE_DIR=analysisFolder,  # Passez le répertoire de base
            imageViewer=imageViewer,
            displayStack=displayStack,
            fileNameLabel=fileNameLabel,
            update_window_button_visibility=update_window_button_visibility,
            update_videoBar_button_visibility=update_videoBar_button_visibility
        )
        return

    fileNameLabel.setText(selected_file)
    update_window_button_visibility(True)

    try:
        if file_path.endswith(".json"):
            with open(file_path, "r") as f:
                content = f.read()
            logViewer.setPlainText(content)
            displayStack.setCurrentWidget(logViewer)
            update_videoBar_button_visibility(False)
        elif file_path.endswith(".png") or file_path.endswith(".jpg"):
            pixmap = QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(
                imageViewer.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            imageViewer.setPixmap(scaled_pixmap)
            displayStack.setCurrentWidget(imageViewer)
            update_videoBar_button_visibility(False)
        elif file_path.endswith(".mp4"):
            mediaPlayer.setSource(QUrl.fromLocalFile(file_path))
            mediaPlayer.play()
            displayStack.setCurrentWidget(videoViewer)
            update_videoBar_button_visibility(True)
        else:
            logViewer.setPlainText(f"❌ Format de fichier non pris en charge : {selected_file}")
            displayStack.setCurrentWidget(logViewer)
            update_videoBar_button_visibility(False)
    except Exception as e:
        print(f"❌ Erreur lors de l'affichage du fichier : {e}")
        logViewer.setPlainText(f"❌ Erreur lors de l'affichage du fichier : {selected_file}")
        displayStack.setCurrentWidget(logViewer)

def calculate_time_estimation(video_duration_seconds, fps, model_name):
    """ Calcule une estimation du temps de traitement en fonction de la durée de la vidéo,
    du fps et du modèle sélectionné. """
    # Temps moyen par frame en secondes pour chaque modèle
    average_time_per_frame = {
        "Yolo11": 0.05,  # Exemple : 50 ms par frame
        "Yolo8": 0.03,   # Exemple : 30 ms par frame
        "Clip": 0.1,     # Exemple : 100 ms par frame
        "Florence-2": 0.08  # Exemple : 80 ms par frame
    }
    # Récupérer le temps moyen par frame pour le modèle sélectionné
    time_per_frame = average_time_per_frame.get(model_name, 0.05)  # Valeur par défaut : 50 ms
    # Calculer le temps total en secondes
    total_time_seconds = video_duration_seconds * fps * time_per_frame
    # Convertir en hh:mm:ss
    hours = int(total_time_seconds // 3600)
    minutes = int((total_time_seconds % 3600) // 60)
    seconds = int(total_time_seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"