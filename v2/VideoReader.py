from  imports import *

def upload_video():
    """Ouvre une boîte de dialogue pour sélectionner une vidéo."""
    filePath, _ = QFileDialog.getOpenFileName(None, "Sélectionner une vidéo", "", "Videos (*.mp4 *.avi *.mov)")
    return filePath

def on_slider_pressed(video_analysis_ui):
    """Indique que l'utilisateur commence à interagir avec le slider."""
    video_analysis_ui.isSeeking = True
    
def on_slider_released(video_analysis_ui):
    """Met à jour la position de la vidéo lorsque l'utilisateur relâche le slider."""
    slider_value = video_analysis_ui.videoSlider.value()  # Récupérer la valeur actuelle du slider
    seek_video(video_analysis_ui.mediaPlayer, slider_value, video_analysis_ui.videoSlider)  # Mettre à jour la position
    video_analysis_ui.isSeeking = False 

def on_video_loaded(video_analysis_ui):
    """Active la videoBar lorsque la vidéo est chargée."""
    video_analysis_ui.videoBarLayout.setEnabled(True)
    # Connecter l'événement positionChanged pour mettre à jour le slider et les temps
    video_analysis_ui.mediaPlayer.positionChanged.connect(
        lambda position: update_video_progress(
            video_analysis_ui.mediaPlayer,
            video_analysis_ui.videoSlider,
            video_analysis_ui.videoTimeLabel,
            video_analysis_ui.isSeeking,
            position
        )
    ) # Réinitialiser le slider lorsque la durée de la vidéo change
    video_analysis_ui.mediaPlayer.durationChanged.connect(
        lambda: video_analysis_ui.videoSlider.setValue(0)
    ) # Remettre le bouton en mode "Play" à la fin de la vidéo
    video_analysis_ui.mediaPlayer.mediaStatusChanged.connect(
        lambda status: reset_play_button_on_end(
            video_analysis_ui.mediaPlayer,
            video_analysis_ui.playPauseButton
        )
    )

def reset_play_button_on_end(media_player, play_pause_button):
    """Remet le bouton en mode 'Play' lorsque la vidéo est terminée."""
    if media_player.mediaStatus() == QMediaPlayer.MediaStatus.EndOfMedia:
        play_pause_button.setText("▶")

def toggle_video_playback(media_player, play_pause_button):
    """Met en pause ou reprend la lecture de la vidéo."""
    if media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
        media_player.pause()
        play_pause_button.setText("▶")  # Changer l'icône en Play
    else:
        media_player.play()
        play_pause_button.setText("⏸")  # Changer l'icône en Pause

def update_video_progress(media_player, video_slider, video_time_label, is_seeking, position):
    """Met à jour le slider et le texte du temps en fonction de la progression de la vidéo."""
    if is_seeking:  
        return  # Bloquer la mise à jour pendant que l'utilisateur interagit avec le slider
    duration = media_player.duration()
    if duration > 0:
        current_time = position // 1000  # Utiliser la position transmise par le signal
        total_time = duration // 1000
        slider_value = int((current_time / total_time) * 100)
        video_slider.setValue(slider_value)  
        video_time_label.setText(f"{format_time(current_time)} / {format_time(total_time)}")

def seek_video(media_player, position, video_slider):
    """Permet de naviguer dans la vidéo en cliquant sur le slider."""
    if media_player.duration() > 0:
        new_position = int((position / 100) * media_player.duration())  # Convertir en entier
        media_player.setPosition(new_position)

def format_time(seconds):
    """Formate le temps en mm:ss."""
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02}:{seconds:02}"