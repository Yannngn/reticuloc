def decode_label(label: int, label_dict: dict[int, str] | None) -> str:
    if label_dict is None:
        return str(label).replace(" ", "_")

    return label_dict[label].replace(" ", "_")
