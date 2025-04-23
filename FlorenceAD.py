import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import cv2
from pathlib import Path
from tqdm import tqdm
import json

class FlorenceActionDetector:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large",
            torch_dtype=self.dtype,
            trust_remote_code=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def analyze_video_with_prompt(self, video_path, prompt, target_fps=0.5):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Impossible d'ouvrir la vidéo: {video_path}")
            return -1, None, None, None

        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = max(1, int(fps / target_fps))
        frame_count = 0

        prompt_embedding = self.embedder.encode(prompt, convert_to_tensor=True)

        best_score = -1
        best_description = None
        best_frame_number = None
        best_frame_image = None

        with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc=f"Analyse de {video_path.name}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                pbar.update(1)

                if frame_count % interval == 0:
                    image_path = "temp.jpg"
                    cv2.imwrite(image_path, frame)
                    image = Image.open(image_path)

                    inputs = self.processor(images=image, return_tensors="pt").to(self.device, self.dtype)
                    generated_ids = self.model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=512,
                        num_beams=3,
                        do_sample=False
                    )
                    description = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                    desc_embedding = self.embedder.encode(description, convert_to_tensor=True)
                    score = util.cos_sim(prompt_embedding, desc_embedding).item()

                    if score > best_score:
                        best_score = score
                        best_description = description
                        best_frame_number = frame_count
                        best_frame_image = frame.copy()

                frame_count += 1

        cap.release()
        return best_score, best_description, best_frame_number, best_frame_image

def main():
    # Paramètres par défaut
    query = "A person walking on an urban roadway "
    video_dir = Path("Video")
    min_score = 0.30

    videos = list(video_dir.glob("*.mp4"))
    if not videos:
        print("❌ Aucune vidéo trouvée dans le dossier.")
        return

    detector = FlorenceActionDetector()
    results = []

    for video_path in videos:
        score, desc, frame, image = detector.analyze_video_with_prompt(video_path, query)
        print(f"\n📊 Vidéo : {video_path.name} | Score : {round(score, 4)}")
        if score >= min_score:
            results.append({
                "score": score,
                "video": video_path.name,
                "description": desc,
                "frame": frame
            })

    if not results:
        print("❌ Aucune correspondance suffisante trouvée.")
        return

    results.sort(key=lambda x: x["score"], reverse=True)
    best = results[0]

    print("\n🎯 Résultat le plus proche :")
    print(f"📁 Vidéo : {best['video']}")
    print(f"🧠 Similarité : {round(best['score'], 4)}")
    print(f"📝 Description générée : {best['description']}")
    print(f"🎞️ Frame correspondante : {best['frame']}")


    # Sauvegarde JSON
    result_path = Path("results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"📁 Résultats sauvegardés dans : {result_path}")

if __name__ == "__main__":
    main()
