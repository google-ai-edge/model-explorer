import type { Extension } from '../common/types';
import type { AdapterExecuteResponse, AdapterOverrideResponse, AdapterStatusCheckResponse, AdapterStatusCheckResults } from '../common/extension_command';

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

/**
 * @deprecated
 * @todo Revert mock API changes!
 */
export function mockOptimizationPolicies(json: Extension[]) {
  if (!isMockEnabled) {
    return;
  }

  json.forEach((ext) => {
    if (ext.id === 'tt_adapter') {
      ext.settings = {
        optimizationPolicies: ['Foo', 'Bar', 'Baz', 'Quux']
      };
    }
  });
}

function processAttribute(key: string, value: string) {
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
    json.graphs?.forEach((graph: { nodes: { attrs: { key: string, value: string }[]}[]}) => {
      graph.nodes?.forEach((node) => {
        node.attrs?.forEach(({key, value}, index) => {
          node.attrs[index] = processAttribute(key, value);
        });
      });
    });

    // json.perf_data = {
    //   'ttir-graph': {
    //     results: {
    //       'forward0': {
    //         value: 1,
    //         bgColor: '#ff0000',
    //         textColor: '#000000'
    //       }
    //     }
    //   }
    // };

    return json;
  }

  if (command === 'status_check') {
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

    return { graphs: [MOCK_STATUS_UPDATE] } satisfies AdapterStatusCheckResponse;
  }

  if (command === 'execute') {
    return {
      graphs: [],
    } satisfies AdapterExecuteResponse;
  }

  if (command === 'override') {
    return {
      graphs: [{
        success: true,
      }],
    } satisfies AdapterOverrideResponse;
  }

  return json;
}
