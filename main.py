"""
Pipeline principal que carga el dataset y el modelo e inicia el entrenamiento de este.

"""



from src.dataset import FashionpediaDataset
from src.model import NUM_CLASSES, get_device, get_model
from src.train import parse_args, train_model



if __name__ == "__main__":

    # lee los argumentos pasados por comandos (epochs, batch size) y detecta si hay GPU
    args = parse_args()
    device = get_device()

    # carga los datasets de train y val/test
    train_dataset = FashionpediaDataset(
        "dataset/train",
        "dataset/instances_train_no_humans.json",
        image_size=args.image_size,
    )
    val_dataset = FashionpediaDataset(
        "dataset/test_no_humans",
        "dataset/instances_test_no_humans.json",
        image_size=args.image_size,
    )

    # carga el modelo preentrenado Mask R-CNN
    model = get_model(num_classes=NUM_CLASSES, pretrained=not args.no_pretrained)
    model.to(device)

    # inicia el entrenamiento
    train_model(
        model,
        train_dataset,
        val_dataset,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        workers=args.workers,
        checkpoint_dir=args.checkpoint_dir,
        max_train_images=args.max_train_images,
        max_val_images=args.max_val_images,
    )
