from src.utils.vision_utils import add_image_padding

def collate_fn(batch, processor, tokenizer):
    """
    Dataset({
        features: ['question_id', 'image', 'question', 'answer', 'id', 'license', 'file_name', 'coco_url', 'height', 'width', 'date_captured'],
        num_rows: 40504
    })
    > dataset[0]
    {
        'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=400x400>,
        'question': 'Please carefully observe the image and come up with a caption for the image.',
        'answer': ['A bicycle replica with a clock as the front wheel.',
                    'The bike has a clock as a tire.',
                    'A black metal bicycle with a clock inside the front wheel.',
                    'A bicycle figurine in which the front wheel is replaced with a clock\n',
                    'A clock with the appearance of the wheel of a bicycle '],
    }
    input format:
    "<|extra_0|><|extra_1|>\n{sample['question']}\n{sample['answer'][0]}<|endoftext|>"
    """
    assert tokenizer.padding_side == "right", "You should use padding_side='right' in your tokenizer"
    FIXED_QUERY_LEN = 256
    captions = []
    images_pad = []
    image_pad_tokens = "".join([tokenizer.pad_token] * FIXED_QUERY_LEN)
    for sample in batch:
        if len(sample['answer']) > 0:
            caption = f"<|extra_0|>{image_pad_tokens}<|extra_1|>\n{sample['question']}\n{sample['answer'][0]}<|endoftext|>"
        else:
            caption = f"<|extra_0|>{image_pad_tokens}<|extra_1|>\n{sample['question']}\n"
        captions.append(caption)
        images_pad.append(add_image_padding(sample['image']))

    # get vision features
    vision_batch_inputs = processor(
        images=images_pad,
        padding="max_length",
        return_tensors="pt")

    # get text inputs
    text_batch_inputs = tokenizer(
        captions,
        padding="longest",
        return_tensors="pt")

    # label
    labels = text_batch_inputs["input_ids"].clone()
    labels[:,1:(FIXED_QUERY_LEN + 1)] = -100

    attention_mask = text_batch_inputs["attention_mask"]
    for idx, label in enumerate(labels):
        pad_mask = (label == tokenizer.pad_token_id)
        pad_exists = pad_mask.any()
        if pad_exists:
            pad_idxs = pad_mask.nonzero(as_tuple=True)[0]
            if len(pad_idxs)==1:
                pass
            else:
                label[pad_idxs[0]+1:] = -100 # keep 1 pad token only, this pad token is eos token
            attention_mask[idx,pad_idxs[0]] = 1
        # new label include eos_token_id
        labels[idx] = label

    return {
        'pixel_values': vision_batch_inputs['pixel_values'],
        'input_ids': text_batch_inputs['input_ids'],
        'attention_mask': attention_mask,
        'labels': labels
    }