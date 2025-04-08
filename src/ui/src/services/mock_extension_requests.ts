import type { AdapterExecuteResponse, AdapterStatusCheckResponse, AdapterStatusCheckResults, ExtensionCommand } from '../common/extension_command.js';
import type { GraphCollection } from '../components/visualizer/common/input_graph.js';
import type { NodeAttribute } from '../custom_element/index.js';

export const isMockEnabled = localStorage.getItem('mock-api') === 'true';

const MOCK_STATUS_UPDATE: Required<Omit<AdapterStatusCheckResults, 'error'>> = {
  isDone: false,
  progress: 0,
  total: 100,
  timeElapsed: 0,
  currentStatus: 'executing',
  stdout: '',
  log_file: '/fake.log'
};


function processAttribute(attr: NodeAttribute): NodeAttribute {
  if (attr.editable) {
    return attr;
  }

  if (typeof attr.value !== 'string') {
    return {
      key: attr.key,
      value: attr.value
    };
  }

  if (attr.key.includes('memory')) {
    return {
      key: attr.key,
      value: attr.value,
      display_type: 'memory'
    }
  }

  if (attr.key.includes('grid')) {
    return {
      key: attr.key,
      value: attr.value,
      editable: {
        input_type: 'grid',
        separator: 'x',
        min_value: 0,
        max_value: 10,
        step: 1
      }
    };
  }

  if (attr.value.startsWith('[')) {
    return {
      key: attr.key,
      value: attr.value,
      editable: {
        input_type: 'int_list',
        min_value: 0,
        max_value: 128,
        step: 32
      }
    };
  }

  if (attr.value.startsWith('(')) {
    return {
      key: attr.key,
      value: attr.value
    };
  }

  return {
    key: attr.key,
    value: attr.value,
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
      node.attrs?.forEach((nodeAttribute, index) => {
        node.attrs![index] = processAttribute(nodeAttribute);
      });

      if (!node.attrs?.find(({ key }) => key.includes('memory'))) {
        node.attrs?.push(processAttribute({ key: 'memory', value: '0.5' }));
      }

      if (!node.attrs?.find(({ key }) => key.includes('grid'))) {
        node.attrs?.push(processAttribute({ key: 'grid', value: '1x1' }));
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


/**
 * @deprecated
 * @todo Revert mock API changes!
 */
export function interceptExtensionCommand(command: ExtensionCommand) {
  if (!isMockEnabled) {
    return undefined;
  }

  if (command.cmdId === 'execute') {
    const responseBody = JSON.stringify({
      graphs: [],
    } satisfies AdapterExecuteResponse);

    return new Response(responseBody, { status: 200 });
  }

  if (command.cmdId === 'status_check') {
    if (MOCK_STATUS_UPDATE.isDone) {
      MOCK_STATUS_UPDATE.isDone = false;
      MOCK_STATUS_UPDATE.progress = 0;
      MOCK_STATUS_UPDATE.currentStatus = 'executing';
      MOCK_STATUS_UPDATE.timeElapsed = 0;
    }

    MOCK_STATUS_UPDATE.timeElapsed = MOCK_STATUS_UPDATE.timeElapsed + Math.trunc(Math.random() * 100);
    MOCK_STATUS_UPDATE.progress += Math.trunc(Math.random() * 10);

    if (MOCK_STATUS_UPDATE.progress >= MOCK_STATUS_UPDATE.total) {
      MOCK_STATUS_UPDATE.isDone = true;
      MOCK_STATUS_UPDATE.progress = MOCK_STATUS_UPDATE.total;
      MOCK_STATUS_UPDATE.currentStatus = 'finished';
    }

    const responseBody = JSON.stringify({ graphs: [MOCK_STATUS_UPDATE] } satisfies AdapterStatusCheckResponse);

    return new Response(responseBody, { status: 200 });
  }

  return undefined;
}
