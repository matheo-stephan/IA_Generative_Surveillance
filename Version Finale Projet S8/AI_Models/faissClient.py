from imports import np, faiss, os, json

class FaissClient:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(self.dimension)  # Index global pour toutes les collections
        self.metadata = {}  # Dictionnaire pour stocker les métadonnées {id: {"collection": ..., "name": ...}}
        self.next_id = 0  # ID unique pour chaque embedding

    def add_to_collection(self, collection_name, ids, embeddings):
        """
        Ajoute des embeddings à une collection spécifique.
        """
        embeddings = np.array(embeddings).astype('float32')
        self.index.add(embeddings)

        for i, frame_name in enumerate(ids):
            self.metadata[self.next_id] = {"collection": collection_name, "name": frame_name}
            self.next_id += 1

        print(f"✅ {len(embeddings)} embeddings ajoutés à la collection '{collection_name}'.")

    def search(self, query_embedding, k=5):
        """
        Recherche les k embeddings les plus proches dans l'index global.
        """
        query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        results = []

        for i, idx in enumerate(indices[0]):
            if idx != -1:
                metadata = self.metadata[idx]
                results.append({
                    "collection": metadata["collection"],
                    "name": metadata["name"],
                    "distance": distances[0][i]
                })

        return results

    def save(self, path):
        """
        Sauvegarde l'index FAISS et les métadonnées dans un fichier.
        """
        # Sauvegarder l'index FAISS
        faiss.write_index(self.index, path)
        print(f"✅ Index FAISS sauvegardé dans '{path}'.")

        # Sauvegarder les métadonnées dans un fichier JSON
        metadata_path = path.replace(".faiss", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f)
        print(f"✅ Métadonnées sauvegardées dans '{metadata_path}'.")

    def load(self, path):
        """
        Charge l'index FAISS et les métadonnées depuis un fichier.
        """
        if os.path.exists(path):
            # Charger l'index FAISS
            self.index = faiss.read_index(path)
            print(f"✅ Index FAISS chargé depuis '{path}'.")

            # Charger les métadonnées depuis un fichier JSON
            metadata_path = path.replace(".faiss", "_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)
                # Convertir les clés en entiers (elles sont sauvegardées comme chaînes dans JSON)
                self.metadata = {int(k): v for k, v in self.metadata.items()}
                print(f"✅ Métadonnées chargées depuis '{metadata_path}'.")

                # Mettre à jour self.next_id pour éviter les collisions d'IDs
                if self.metadata:
                    self.next_id = max(self.metadata.keys()) + 1
                else:
                    self.next_id = 0
            else:
                print(f"⚠️ Aucun fichier de métadonnées trouvé à '{metadata_path}'.")
        else:
            print(f"⚠️ Aucun fichier FAISS trouvé à '{path}'. Un nouvel index sera créé.")

    def display_collections(self):
        """
        Affiche les collections et le nombre de frames/embeddings dans chacune.
        """
        if not self.metadata:
            print("⚠️ Aucune collection disponible dans la base de données.")
            return

        collections = {}
        for metadata in self.metadata.values():
            collection_name = metadata["collection"]
            collections[collection_name] = collections.get(collection_name, 0) + 1

        print("🔍 Contenu de la base de données FAISS :")
        for collection_name, count in collections.items():
            print(f" - Collection : {collection_name}, Nombre de frames/embeddings : {count}")

    def reset_database(self, path):
        """
        Réinitialise la base de données FAISS (avec confirmation) et met à jour les fichiers sur le disque.
        """
        confirmation = input("⚠️ Êtes-vous sûr de vouloir réinitialiser la base de données ? (oui/non) : ")
        if confirmation.lower() == "oui":
            self.index = faiss.IndexFlatL2(self.dimension)  # Réinitialiser l'index
            self.metadata = {}  # Réinitialiser les métadonnées
            self.next_id = 0  # Réinitialiser l'ID

            # Supprimer les fichiers existants
            faiss_path = path
            metadata_path = path.replace(".faiss", "_metadata.json")
            if os.path.exists(faiss_path):
                os.remove(faiss_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)

            print("✅ Base de données FAISS réinitialisée et fichiers supprimés.")
        else:
            print("❌ Réinitialisation annulée.")

    def delete_collection(self, collection_name):
        """
        Supprime une collection spécifique de la base de données FAISS.
        """
        # Filtrer les IDs appartenant à la collection
        ids_to_remove = [id for id, meta in self.metadata.items() if meta["collection"] == collection_name]

        if not ids_to_remove:
            print(f"⚠️ Aucune collection trouvée avec le nom '{collection_name}'.")
            return

        # Supprimer les embeddings correspondants de l'index
        self.index.remove_ids(np.array(ids_to_remove, dtype=np.int64))

        # Supprimer les métadonnées associées
        for id in ids_to_remove:
            del self.metadata[id]

        print(f"✅ Collection '{collection_name}' supprimée avec succès.")