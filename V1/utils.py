import os
from PyQt6.QtWidgets import QFileDialog, QTreeWidgetItem, QMessageBox
import shutil
import zipfile

def upload_video():
    """Ouvre une boîte de dialogue pour sélectionner une vidéo."""
    filePath, _ = QFileDialog.getOpenFileName(None, "Sélectionner une vidéo", "", "Videos (*.mp4 *.avi *.mov)")
    return filePath

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
