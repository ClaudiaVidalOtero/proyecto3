from pathlib import Path
import zipfile
import shutil

from huggingface_hub import snapshot_download


# Repositorio de DeepFashion2
REPO_ID = "sahirp/deepfashion2"


def extract_zip(zip_path: Path, output_dir: Path):
    """Descomprime un archivo ZIP."""

    print(f"\n[EXTRACT] {zip_path.name}")

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(output_dir)

    print(f"[OK] {zip_path.name} descomprimido.")


def main():

    print("=" * 70)
    print("       DEEPFASHION2 - HUGGING FACE DOWNLOADER")
    print("=" * 70)

    print()
    print("Indica la carpeta donde quieres guardar el dataset.")
    print("Ejemplos:")
    print(r"  Windows: D:\datasets\deepfashion2")
    print(r"  Windows: C:\Users\TuUsuario\Desktop\dataset")
    print()

    output = input("Ruta de destino: ").strip()

    # Quitar comillas si el usuario las ha puesto
    output = output.strip('"').strip("'")

    if not output:
        print("\n[ERROR] No has indicado ninguna ruta.")
        return

    output_dir = Path(output).expanduser().resolve()

    # Crear carpeta
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 70)
    print("CONFIGURACIÓN")
    print("=" * 70)

    print(f"Repositorio : {REPO_ID}")
    print(f"Destino     : {output_dir}")

    # ---------------------------------------------------------
    # Confirmación
    # ---------------------------------------------------------

    print()

    confirm = input(
        "¿Continuar con esta carpeta? [S/n]: "
    ).strip().lower()

    if confirm == "n":
        print("Cancelado.")
        return

    # ---------------------------------------------------------
    # DESCARGAR REPOSITORIO
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("DESCARGANDO DESDE HUGGING FACE")
    print("=" * 70)

    print()
    print("La descarga puede tardar bastante.")
    print("No cierres el programa mientras esté descargando.")
    print()

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(output_dir),
    )

    print()
    print("[OK] Descarga completada.")

    # ---------------------------------------------------------
    # BUSCAR ZIP
    # ---------------------------------------------------------

    zip_files = list(output_dir.glob("*.zip"))

    if zip_files:

        print()
        print("=" * 70)
        print("DESCOMPRIMIENDO")
        print("=" * 70)

        for zip_path in zip_files:

            extract_zip(
                zip_path,
                output_dir
            )

        # -----------------------------------------------------
        # BORRAR ZIP
        # -----------------------------------------------------

        print()
        print("Los ZIP ocupan mucho espacio.")

        delete_zips = input(
            "¿Quieres eliminar los ZIP después de extraerlos? [S/n]: "
        ).strip().lower()

        if delete_zips != "n":

            for zip_path in zip_files:

                try:
                    zip_path.unlink()

                    print(
                        f"[DELETE] {zip_path.name}"
                    )

                except Exception as e:

                    print(
                        f"[WARNING] No se pudo eliminar "
                        f"{zip_path.name}: {e}"
                    )

    else:

        print()
        print("[INFO] No se encontraron ZIP para descomprimir.")

    # ---------------------------------------------------------
    # MOSTRAR RESULTADO
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("DATASET LISTO")
    print("=" * 70)

    print()
    print(f"Dataset guardado en:")

    print()
    print(f"  {output_dir}")

    print()
    print("Contenido:")

    for item in sorted(output_dir.iterdir()):

        if item.is_dir():

            print(f"  📁 {item.name}")

        else:

            print(f"  📄 {item.name}")

    print()
    print("=" * 70)
    print("FIN")
    print("=" * 70)


if __name__ == "__main__":
    main()
