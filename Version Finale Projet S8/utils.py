from imports import *
from AI_Models.faiss_instance import faiss_client

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Root folder
LOGO_PATH = os.path.join(BASE_DIR, "assets/logo.png")  # Logo path

def load_hierarchy(tree_widget, directory):
    """Load files and folders into the given QTreeWidget, with 'Frames' folder displayed first."""
    tree_widget.clear()
    items = []

    # Parcourir les fichiers et dossiers
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        tree_item = QTreeWidgetItem([item])
        if os.path.isdir(item_path):
            tree_item.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
            # Priorité pour le dossier Frames
            priority = 0 if item == "Frames" else 1
            items.append((priority, tree_item, item_path))
        else:
            # Priorité moindre pour les fichiers
            items.append((2, tree_item, item_path))

    # Trier les éléments : Frames en premier, puis autres dossiers, puis fichiers
    items.sort(key=lambda x: x[0])

    # Ajouter les éléments triés au QTreeWidget
    for _, tree_item, item_path in items:
        tree_widget.addTopLevelItem(tree_item)
        # Si c'est un dossier, charger son contenu
        if os.path.isdir(item_path):
            populate_tree(item_path, tree_item)

def populate_tree(directory, parent_item):
    """Populate QTreeWidget with files and folders from a repository."""
    for item in sorted(os.listdir(directory)):
        item_path = os.path.join(directory, item)
        tree_item = QTreeWidgetItem(parent_item, [item])
        if os.path.isdir(item_path):
            tree_item.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
            populate_tree(item_path, tree_item)

def get_selected_item_path(tree_widget, base_dir):
    """Get the full path of the selected item in the tree widget."""
    if tree_widget is None:
        QMessageBox.warning(None, "Warning", "No active tree widget.")
        return None

    selected_item = tree_widget.currentItem()
    if not selected_item:
        QMessageBox.warning(tree_widget, "Warning", "No file or folder selected.")
        return None
    
    # Determine the subdirectory (Bank or Analysis) based on the tree_widget's objectName
    if tree_widget.objectName() == "bankNav":
        sub_dir = "Bank"
    elif tree_widget.objectName() == "fileNav":
        sub_dir = "Analysis"
    else:
        QMessageBox.critical(tree_widget, "Error", "Unknown tree widget.")
        return None

    # Build the full path
    file_path = selected_item.text(0)
    current_item = selected_item
    while current_item.parent() is not None:
        current_item = current_item.parent()
        file_path = os.path.join(current_item.text(0), file_path)
    file_path = os.path.join(base_dir, sub_dir, file_path)

    if not os.path.exists(file_path):
        QMessageBox.critical(tree_widget, "Error", f"File or folder '{file_path}' doesn't exist.")
        return None

    return file_path

def release_file_resources(file_path, mediaPlayer=None):
    """Release resources associated with a file."""
    try:
        if mediaPlayer is None:
            print("❌ mediaPlayer is None. Cannot release resources.")
            return
        # If a media player is provided, stop it and clear the source
        if mediaPlayer.mediaStatus() != QMediaPlayer.MediaStatus.NoMedia:
            mediaPlayer.stop()
            mediaPlayer.setSource(QUrl())  # Clear the media source
            print(f"🔄 Media player resources released for: {file_path}")

        # Attempt to open the file in read mode to ensure it's not locked
        with open(file_path, 'rb'):
            pass
        print(f"✅ File resources released for: {file_path}")
    except Exception as e:
        print(f"❌ Failed to release resources for {file_path}: {e}")

def rename_item(tree_widget, base_dir, mediaPlayer=None, fileNameLabel=None):
    """Rename the selected file or folder while preserving the file extension."""
    file_path = get_selected_item_path(tree_widget, base_dir)
    if not file_path:
        return

    # Extract the current name and extension
    old_name = os.path.basename(file_path)
    name, ext = os.path.splitext(old_name)  # Separate the name and extension

    while True:  # Loop until a valid name is provided
        # Show the custom rename dialog
        new_name, ok = QInputDialog.getText(
            None,
            "Rename",
            f"Enter new name (extension '{ext}' will be preserved):",
            QLineEdit.EchoMode.Normal,
            name
        )
        if not ok or not new_name.strip():
            return  # User canceled or entered an empty name

        # Rebuild the new path with the preserved extension
        new_name = new_name.strip() + ext
        new_path = os.path.join(os.path.dirname(file_path), new_name)

        # Check if a file or folder with the new name already exists
        if os.path.exists(new_path):
            QMessageBox.warning(None, "Error", f"'{new_name}' already exists. Please try again.")
            continue  # Reopen the dialog for another attempt

        try:
            # Release resources associated with the file
            release_file_resources(file_path, mediaPlayer)

            # Perform the rename operation
            os.rename(file_path, new_path)
            tree_widget.currentItem().setText(0, new_name)  # Update the tree widget
            if fileNameLabel:
                fileNameLabel.setText(new_name)
            QMessageBox.information(None, "Success", f"Renamed to '{new_name}'.")
            break  # Exit the loop after a successful rename
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error while renaming: {e}")
            return

def delete_item(tree_widget, base_dir, mediaPlayer, display_logo=None, *args):
    """Delete content after confirmation."""
    file_path = get_selected_item_path(tree_widget, base_dir)
    if not file_path:
        return

    # Ask for confirmation before deleting
    reply = QMessageBox.question(
        tree_widget,
        "Confirmation",
        f"Are you sure you want to delete '{os.path.basename(file_path)}' and all its content?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )

    if reply == QMessageBox.StandardButton.Yes:
        try:
            # Release resources associated with the file
            release_file_resources(file_path, mediaPlayer)

            # Handle reciprocal deletion for bankNav
            if tree_widget.objectName() == "bankNav":
                # Check if the item is a video
                if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    # Find and delete the associated video_frames folder
                    frames_dir = os.path.join(
                        os.path.dirname(file_path),
                        "Frames",
                        os.path.splitext(os.path.basename(file_path))[0] + "_frames"
                    )
                    if os.path.exists(frames_dir):
                        shutil.rmtree(frames_dir)
                        print(f"🗑️ Associated folder '{frames_dir}' deleted.")

                    # Delete the associated collection in FAISS
                    if faiss_client:
                        collection_name = os.path.splitext(os.path.basename(file_path))[0]
                        print(f"🗑️ Deleting collection '{collection_name}' from FAISS.")
                        faiss_client.delete_collection(collection_name)
                    elif faiss_client is None:
                        print("❌ FAISS client is None. Cannot delete collection.")

                # Check if the item is a video_frames folder
                elif os.path.basename(file_path).endswith("_frames") and os.path.isdir(file_path):
                    # Find and delete the associated video file
                    video_name = os.path.basename(file_path).replace("_frames", "")
                    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
                    for ext in video_extensions:
                        video_path = os.path.join(os.path.dirname(file_path), video_name + ext)
                        if os.path.exists(video_path):
                            os.remove(video_path)
                            print(f"🗑️ Associated video '{video_path}' deleted.")
                            break

            # Delete the file or folder
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

            # Remove the item from the tree widget
            selected_item = tree_widget.currentItem()
            parent = selected_item.parent()
            if parent:
                parent.removeChild(selected_item)
            else:
                tree_widget.takeTopLevelItem(tree_widget.indexOfTopLevelItem(selected_item))
            
            # Reload the tree widget based on its objectName
            if tree_widget.objectName() == "bankNav":
                bank_dir = os.path.join(base_dir, "Bank")
                load_hierarchy(tree_widget, bank_dir)
            elif tree_widget.objectName() == "fileNav":
                analysis_dir = os.path.join(base_dir, "Analysis")
                load_hierarchy(tree_widget, analysis_dir)
            QMessageBox.information(tree_widget, "Success", f"'{os.path.basename(file_path)}' has been deleted.")

            # Optionally display the logo after deletion
            if display_logo:
                display_logo()
        except Exception as e:
            QMessageBox.critical(tree_widget, "Error", f"Error while deleting: {e}")

def export_item(tree_widget, base_dir):
    """Export selected element with QTreeWidget under ZIP format."""
    file_path = get_selected_item_path(tree_widget, base_dir)
    if not file_path:
        print("❌ No file or folder selected for export.")
        return

    zip_path, _ = QFileDialog.getSaveFileName(None, "Export under ZIP format", "", "ZIP Files (*.zip)")
    if zip_path:
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                if os.path.isdir(file_path):
                    for root, _, files in os.walk(file_path):
                        for file in files:
                            file_path_full = os.path.join(root, file)
                            arcname = os.path.relpath(file_path_full, file_path)
                            zipf.write(file_path_full, arcname)
                else:
                    zipf.write(file_path, os.path.basename(file_path))
            QMessageBox.information(None, "Success", f"Export successful: {zip_path}")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error while exporting: {e}")

def display_previous_file(tree_widget, display_file, logViewer, displayStack, mediaPlayer, videoViewer, imageViewer, fileNameLabel, update_window_button_visibility, update_videoBar_button_visibility):
    """Display the previous file in the tree widget."""
    if tree_widget is None:
        print("❌ No active tree widget provided.")
        return

    current_item = tree_widget.currentItem()
    if current_item:
        prev_item = tree_widget.itemAbove(current_item)
        if prev_item:
            tree_widget.setCurrentItem(prev_item)
            # Determine the base directory dynamically
            base_dir = tree_widget.objectName() == "bankNav" and "Bank" or "Analysis"
            display_file(
                prev_item,
                os.path.join(BASE_DIR, base_dir),
                logViewer,
                displayStack,
                mediaPlayer,
                videoViewer,
                imageViewer,
                fileNameLabel,
                update_window_button_visibility,
                update_videoBar_button_visibility
            )
        else:
            print("❌ No previous file.")

def display_next_file(tree_widget, display_file, logViewer, displayStack, mediaPlayer, videoViewer, imageViewer, fileNameLabel, update_window_button_visibility, update_videoBar_button_visibility):
    """Display the next file in the tree widget."""
    if tree_widget is None:
        print("❌ No active tree widget provided.")
        return

    current_item = tree_widget.currentItem()
    if current_item:
        next_item = tree_widget.itemBelow(current_item)
        if next_item:
            tree_widget.setCurrentItem(next_item)
            # Determine the base directory dynamically
            base_dir = tree_widget.objectName() == "bankNav" and "Bank" or "Analysis"
            display_file(
                next_item,
                os.path.join(BASE_DIR, base_dir),
                logViewer,
                displayStack,
                mediaPlayer,
                videoViewer,
                imageViewer,
                fileNameLabel,
                update_window_button_visibility,
                update_videoBar_button_visibility
            )
        else:
            print("❌ No next file.")

def toggle_navigation(splitter, toggleNavButton, adjust_splitter_proportions):
    """Toggle "<" ">" button state"""
    navWidget = splitter.widget(0)  # Connect with it's widget
    if navWidget.isVisible():
        navWidget.hide()  # Hide pannel
        toggleNavButton.setText(">")
        adjust_splitter_proportions(hidden=True)
    else:
        navWidget.show()  # Show pannel
        toggleNavButton.setText("<")
        adjust_splitter_proportions(hidden=False)

def on_file_selected(fileNav, analysisFolder, displayStack, mediaPlayer, videoViewer, jsonViewer, imageViewer):
    """Display content in the right layout depending on file type"""
    selected_item = fileNav.currentItem()
    if not selected_item:
        return
    file_path = os.path.join(analysisFolder, selected_item.text(0))  # Build full path
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
    videoSlider.setVisible(visible)
    playPauseButton.setVisible(visible)
    videoTimeLabel.setVisible(visible)

def update_window_button_visibility(closeButton, prevButton, nextButton, fileNameLabel, deleteButton, exportButton, renameButton, visible):
    closeButton.setVisible(visible)
    prevButton.setVisible(visible)
    nextButton.setVisible(visible)
    fileNameLabel.setVisible(visible)
    renameButton.setVisible(visible)
    deleteButton.setVisible(visible)
    exportButton.setVisible(visible)

def display_logo(imageViewer, displayStack, fileNameLabel, update_window_button_visibility, update_videoBar_button_visibility, mediaPlayer=None, toggleReadmeButton=None):
    """Display logo and close all media"""
    # Stop file reading and free ressources if multimedia monitor is given
    if mediaPlayer:
        mediaPlayer.stop()
        mediaPlayer.setSource(QUrl())
        print("🔄 Reading paused, ressources setted free.")

    # Use logo global path
    print(f"🔍 Logo path : {LOGO_PATH}") # Check for logo path

    if not os.path.exists(LOGO_PATH):
        print("❌ logo.png dosen't exist.")
        return

    pixmap = QPixmap(LOGO_PATH)
    if pixmap.isNull():
        print("❌ Impossible to load logo.png. Check format.")
        return

    scaled_pixmap = pixmap.scaled(
        150, 150,  # Logo size
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation
    )
    imageViewer.setPixmap(scaled_pixmap)
    displayStack.setCurrentWidget(imageViewer)
    fileNameLabel.setText("")  # Erase file name
    update_window_button_visibility(False)  # Mask video buttons
    update_videoBar_button_visibility(False)  # Mask video bar 
    if toggleReadmeButton:
        toggleReadmeButton.setVisible(True)
        toggleReadmeButton.setText("ℹ️")

def display_file(item, base_dir, logViewer, displayStack, mediaPlayer, videoViewer, imageViewer, fileNameLabel, update_window_button_visibility, update_videoBar_button_visibility, toggleReadmeButton=None):
    """Display file content in right layout by default if the file is not readable."""
    if toggleReadmeButton:
        toggleReadmeButton.setVisible(False)  # Hide the toggle button for README

    if item is None or not isinstance(item, QTreeWidgetItem):
        print("❌ No valid element selected. Logo displayed by default.")
        return

    selected_file = item.text(0)
    print(f"🔍 Selected file : {selected_file}")

    file_path = selected_file
    current_item = item
    while current_item.parent() is not None:
        current_item = current_item.parent()
        file_path = os.path.join(current_item.text(0), file_path)
    file_path = os.path.join(base_dir, file_path)

    if os.path.isdir(file_path):
        # If the selected item is a folder, display 📂 in a QLabel
        folder_label = QLabel("📂")
        folder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Center the emoji
        folder_label.setStyleSheet("font-size: 100px;")  # Increase the size of the emoji
        displayStack.addWidget(folder_label)  # Add the QLabel to the stack
        displayStack.setCurrentWidget(folder_label)  # Set it as the current widget
        fileNameLabel.setText(selected_file)
        update_window_button_visibility(True)
        update_videoBar_button_visibility(False)
        return

    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        print(f"❌ File not found : {file_path}")
        display_logo(
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
        elif file_path.endswith(".txt"):
            with open(file_path, "r") as f:
                content = f.read()
            logViewer.setPlainText(content)
            displayStack.setCurrentWidget(logViewer)
            update_videoBar_button_visibility(False)
        else:
            logViewer.setPlainText(f"❌ File type not supported : {selected_file}")
            displayStack.setCurrentWidget(logViewer)
            update_videoBar_button_visibility(False)
    except Exception as e:
        print(f"❌ Error while displaying selected file : {e}")
        logViewer.setPlainText(f"❌ Error while displaying selected file : {selected_file}")
        displayStack.setCurrentWidget(logViewer)


def display_readme(base_dir, logViewer, displayStack, fileNameLabel, update_window_button_visibility, update_videoBar_button_visibility, toggleReadmeButton=None):
    """Display the content of README.html in the logViewer."""
    readme_path = os.path.join(base_dir, "README.html")
    if os.path.exists(readme_path):
        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()
            logViewer.setHtml(content)  # Directly set HTML content
            displayStack.setCurrentWidget(logViewer)
            fileNameLabel.setText("README.html")
            update_window_button_visibility(False)  # Hide buttons
            update_videoBar_button_visibility(False)  # Hide video controls
            if toggleReadmeButton:
                toggleReadmeButton.setVisible(True)
                toggleReadmeButton.setText("❌")
            print("✅ README.html displayed successfully.")
        except Exception as e:
            print(f"❌ Error while displaying README.html: {e}")
    else:
        print("❌ README.html not found.")
