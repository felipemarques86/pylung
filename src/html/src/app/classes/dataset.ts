export interface Dataset {
    name: string;
    image_size: string;
    consensus_level: string;
    pad: string;
    part_size: string;
    type: string;
    num_parts: string;
    description: string;

    count: number;

    deflated: string;

    starting_from: string;
}

export interface DatasetList {
    directory: string;
    datasets: Dataset[];
}


export interface Prediction {
    predicted: number[];
    binary: number;
    predicted_int: number[];
    annotation: string;
    transformed_annotation: string;
    timespent: number;
}

export interface Trial {
    trial_id: string,
    model_type: string,
    image_size: number,
    batch_size: number,
    epochs: number,
    num_classes: number,
    loss: string,
    code_name: string,
    save_weights: boolean,
    static_params: boolean,
    score: number[],
    score_names: string[],
    x_train_size: number,
    y_train_size: number,
    x_valid_size: number,
    y_valid_size: number,
    image_channels: number,
    version: number,
    data_transformer_name: string,
    history: TrialHistory,
    learning_params: LearningParams,
    isolate_nodule_image: boolean,
    detection: boolean
}

export interface LearningParams {
    learning_rate: number,
    projection_dim: number,
    num_heads: number,
    drop_out_1: number,
    drop_out_2: number,
    drop_out_3: number,
    transformer_layers: number,
    patch_size: number,
    activation: string,
    weight: number,
    momentum: number,
    optimizer: string
}

export interface TrialHistory {

    loss: number [],
    accuracy: number[],
    false_positives: number[],
    false_negatives: number[],
    true_negatives: number[],
    true_positives: number[],
    val_loss: number[],
    val_accuracy: number[],
    val_false_positives: number[],
    val_false_negatives: number[],
    val_true_negatives: number[],
    val_true_positives: number[]
}
