import type { Attribute, GraphCollection } from '../components/visualizer/common/input_graph.js';

export const isMockEnabled = localStorage.getItem('mock-api') === 'true';

function processAttribute(key: string, value: string): Attribute {
  if (key.includes('memory')) {
    return {
      key,
      value,
      display_type: 'memory'
    }
  }

  if (key.includes('grid') || key.includes('shape')) {
    return {
      key,
      value,
      editable: {
        input_type: 'grid',
        separator: 'x',
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
 * @todo Revert mock API changes
 */
export function mockGraphCollectionAttributes<T extends GraphCollection>(json: T) {
  json.graphs?.forEach((graph) => {
    graph.nodes?.forEach((node) => {
      node.attrs?.forEach(({key, value}, index) => {
        node.attrs![index] = processAttribute(key, value);
      });

      if (!node.attrs?.find(({ key }) => key.includes('memory'))) {
        node.attrs?.push(processAttribute('memory', '0.5'));
      }
    });

    if (!graph.overlays) {
      graph.overlays = {};
    }
  });

  return json;
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
    return mockGraphCollectionAttributes(json);
  }

  return json;
}
