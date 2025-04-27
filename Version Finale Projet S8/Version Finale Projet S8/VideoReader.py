from  imports import *

def on_slider_pressed(video_analysis_ui):
    """Indicate when user starts interacting with the slider."""
    video_analysis_ui.isSeeking = True
    
def on_slider_released(video_analysis_ui):
    """Update sliders position when mouse click is drop."""
    slider_value = video_analysis_ui.videoSlider.value()  # Take sliders value
    seek_video(video_analysis_ui.mediaPlayer, slider_value, video_analysis_ui.videoSlider)  # Update position
    video_analysis_ui.isSeeking = False 

def on_video_loaded(video_analysis_ui):
    """Activate videoBar when video is uploaded."""
    video_analysis_ui.videoBarLayout.setEnabled(True)
    # Connect positionChanged event to update slider and time text
    video_analysis_ui.mediaPlayer.positionChanged.connect(
        lambda position: update_video_progress(
            video_analysis_ui.mediaPlayer,
            video_analysis_ui.videoSlider,
            video_analysis_ui.videoTimeLabel,
            video_analysis_ui.isSeeking,
            position
        )
    ) # Set slider back to 0 when a new video is played
    video_analysis_ui.mediaPlayer.durationChanged.connect(
        lambda: video_analysis_ui.videoSlider.setValue(0)
    ) # Set "Play" button back to play mode at the end of the vidéo
    video_analysis_ui.mediaPlayer.mediaStatusChanged.connect(
        lambda status: reset_play_button_on_end(
            video_analysis_ui.mediaPlayer,
            video_analysis_ui.playPauseButton
        )
    )

def reset_play_button_on_end(media_player, play_pause_button):
    """Set back bouton mode to 'Play' when video is finished."""
    if media_player.mediaStatus() == QMediaPlayer.MediaStatus.EndOfMedia:
        play_pause_button.setText("▶")

def toggle_video_playback(media_player, play_pause_button):
    """Toggle play and pause video."""
    if media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
        media_player.pause()
        play_pause_button.setText("▶")
    else:
        media_player.play()
        play_pause_button.setText("⏸")

def update_video_progress(media_player, slider, time_label, is_seeking, position):
    """
    Met à jour la progression de la vidéo dans le slider et l'étiquette de temps.
    """
    total_time = media_player.duration()  # Durée totale de la vidéo
    current_time = position  # Position actuelle de la vidéo

    if total_time == 0:  # Éviter la division par zéro
        print("⚠️ Warning: Total video duration is zero. Cannot update progress.")
        return

    slider_value = int((current_time / total_time) * 100)
    slider.setValue(slider_value)

    # Mettre à jour l'étiquette de temps
    current_minutes, current_seconds = divmod(current_time // 1000, 60)
    total_minutes, total_seconds = divmod(total_time // 1000, 60)
    time_label.setText(f"{current_minutes:02}:{current_seconds:02} / {total_minutes:02}:{total_seconds:02}")

def seek_video(media_player, position, video_slider):
    """Navigate through video by clicking on slider's position."""
    if media_player.duration() > 0:
        new_position = int((position / 100) * media_player.duration())  # Convert to int
        media_player.setPosition(new_position)

def format_time(seconds):
    """Time format mm:ss."""
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02}:{seconds:02}"