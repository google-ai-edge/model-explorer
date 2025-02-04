export const isMockEnabled = localStorage.getItem('mock-api') === 'true';

function processAttribute(key: string, value: string) {
  if (key.includes('grid') || key.includes('shape')) {
    return {
      key,
      value,
      editable: {
        input_type: 'grid',
        visual_separator: 'x',
        min_value: 0,
        max_value: 10,
        step: 1
      }
    };
  }

  if (value.startsWith('[')) {
    const arr = value.split(',');

    return {
      key,
      value,
      editable: {
        input_type: 'int_list',
        min_size: 1,
        max_size: arr.length,
        min_value: 0,
        max_value: 128,
        step: 32
      }
    };
  }

  if (value.startsWith('(')) {
    return { key, value };
  }

  return {
    key,
    value,
    editable: {
      input_type: 'value_list',
      options: ['foo', 'bar', 'baz']
    }
  };
}

/**
 * @deprecated
 * @todo Revert mock API changes!
 */
export function mockExtensionCommand(command: string, json: any) {
  if (!isMockEnabled) {
    return json;
  }

  if (command === 'convert') {
    json.graphs?.forEach((graph: { nodes: { attrs: { key: string, value: string }[]}[], overlays: any }) => {
      graph.nodes?.forEach((node) => {
        node.attrs?.forEach(({key, value}, index) => {
          node.attrs[index] = processAttribute(key, value);
        });
      });

      if (!graph.overlays) {
        graph.overlays = {};
      }
    });

    return json;
  }

  return json;
}
