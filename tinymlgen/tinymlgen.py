#!/usr/bin/env python
# -*- coding: utf8 -*-

import re
import hexdump
from datetime import datetime
import tensorflow as tf

def export_c_code(model, optimize=True, model_name="model_data", pretty_print=False):
    """_summary_

    Args:
        model (keras.src.engine.sequential.Sequential): Model instance.
        optimize (bool, optional): Optimization flag. Defaults to False.
        variable_name (str, optional): Model variable name. Defaults to 'model_data'.
        pretty_print (bool, optional): Pretty prints. Defaults to False.
    """
    # Params
    tf_num_ops = 2
    tf_num_inputs = model.input_shape[1]
    tf_num_outputs = model.output_shape[1]

    # Create converter from Keras.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Optimize if flag is set.
    if optimize:
        if isinstance(optimize, bool):
            optimizers = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        else:
            optimizers = optimize
        converter.optimizations = optimizers

    # Convert the model.
    tflite_model = converter.convert()

    # Split bytes by space.
    model_bytes = hexdump.dump(tflite_model).split(" ")

    # Make it like C hex constant value.
    bytes_array = ", ".join(["0x%02x" % int(model_byte, 16) for model_byte in model_bytes])

    # Make C array.
    c_array = "/** Model bytes. */\r\n"
    c_array += "const unsigned char %s[] DATA_ALIGN_ATTRIBUTE = {%s};" % (model_name, bytes_array)
    if pretty_print:
        c_array = c_array.replace("{", "{\n\t").replace("}", "\n}")
        c_array = re.sub(r"(0x..?, ){12}", lambda x: "%s\n\t" % x.group(0), c_array)

    # Get model size.
    model_bytes_size = len(model_bytes)

    # Get the current date and time
    now = datetime.now()

    # Format the date and time in a human-readable way
    formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S")

    # Preamble text.
    preamble = f"""
/***************************************************************************
 * WARNING: This is an automatically generated file.
 * DO NOT MODIFY THIS FILE DIRECTLY.
 * Any changes made to this file will be overwritten
 * when the file is regenerated. To make modifications,
 * please edit the source files that generate this code.
 *
 * File generated on: {formatted_datetime}
****************************************************************************/

#pragma once

#ifdef __has_attribute
#define HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define HAVE_ATTRIBUTE(x) 0
#endif

#if HAVE_ATTRIBUTE(aligned) || (defined(__GNUC__) && !defined(__clang__))
#define DATA_ALIGN_ATTRIBUTE __attribute__((aligned(4)))
#else
#define DATA_ALIGN_ATTRIBUTE
#endif

#define TF_NUM_OPS {tf_num_ops}
#define TF_NUM_INPUTS {tf_num_inputs}
#define TF_NUM_OUTPUTS {tf_num_outputs}

/** Model bytes size. */
const int {model_name}_size = {model_bytes_size};

"""

    # Put all together.
    preamble += c_array

    # Put final new line.
    preamble += "\r\n"

    return preamble
