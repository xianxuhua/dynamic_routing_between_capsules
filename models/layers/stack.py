import tensorflow as tf
tf = tf.compat.v1

def stack(inputs, layer, stack_args, **kwargs):
  """Builds a stack of layers by applying layer repeatedly using stack_args.
  `stack` allows you to repeatedly apply the same operation with different
  arguments `stack_args[i]`. For each application of the layer, `stack` creates
  a new scope appended with an increasing number. For example:
  ```python
    y = stack(x, fully_connected, [32, 64, 128], scope='fc')
    # It is equivalent to:
    x = fully_connected(x, 32, scope='fc/fc_1')
    x = fully_connected(x, 64, scope='fc/fc_2')
    y = fully_connected(x, 128, scope='fc/fc_3')
  ```
  If the `scope` argument is not given in `kwargs`, it is set to
  `layer.__name__`, or `layer.func.__name__` (for `functools.partial`
  objects). If neither `__name__` nor `func.__name__` is available, the
  layers are called with `scope='stack'`.
  Args:
    inputs: A `Tensor` suitable for layer.
    layer: A layer with arguments `(inputs, *args, **kwargs)`
    stack_args: A list/tuple of parameters for each call of layer.
    **kwargs: Extra kwargs for the layer.
  Returns:
    A `Tensor` result of applying the stacked layers.
  Raises:
    ValueError: If the op is unknown or wrong.
  """
  scope = kwargs.pop('scope', None)
  if not isinstance(stack_args, (list, tuple)):
    raise ValueError('stack_args need to be a list or tuple')
  with tf.variable_scope(scope, 'Stack', [inputs]):
    if scope is None:
      if hasattr(layer, '__name__'):
        scope = layer.__name__
      elif hasattr(layer, 'func') and hasattr(layer.func, '__name__'):
        scope = layer.func.__name__  # In case layer is a functools.partial.
      else:
        scope = 'stack'
    outputs = inputs
    for i in range(len(stack_args)):
      # kwargs['scope'] = scope + '_' + str(i + 1)
      layer_args = stack_args[i]
      if not isinstance(layer_args, (list, tuple)):
        layer_args = [layer_args]
      outputs = layer(outputs, *layer_args, **kwargs)
    return outputs
