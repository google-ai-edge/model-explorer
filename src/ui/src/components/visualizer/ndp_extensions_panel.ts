import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  output,
  Signal
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import {MatTooltipModule} from '@angular/material/tooltip';

import {MatDialog} from '@angular/material/dialog';
import {MatSlideToggleModule} from '@angular/material/slide-toggle';
import {MatSnackBar} from '@angular/material/snack-bar';
import {
  ExtensionType,
  NodeDataProviderExtension
} from '../../common/types';
import {ExtensionService} from '../../services/extension_service';
import {AppService} from './app_service';
import {NodeDataProviderExtensionService} from './node_data_provider_extension_service';
import {
  RunNdpExtensionDialog,
  RunNdpExtensionDialogData,
  RunNdpExtensionDialogResult,
} from './run_ndp_extension_dialog';

let ndpId = 0;

/** The drop down menu for add per-node data. */
@Component({
  standalone: true,
  selector: 'ndp-extensions-panel',
  imports: [
    CommonModule,
    MatButtonModule,
    MatIconModule,
    MatProgressSpinnerModule,
    MatSlideToggleModule,
    MatTooltipModule,
  ],
  templateUrl: 'ndp_extensions_panel.ng.html',
  styleUrls: ['./ndp_extensions_panel.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class NodeDataProviderExtensionsPanel {
  readonly onRunDialogRunClicked = output<void>();

  protected readonly extensions: Signal<NodeDataProviderExtension[]> = computed(
    () =>
      this.extensionService.extensions.filter(
        (ext) => ext.type === ExtensionType.NODE_DATA_PROVIDER
      )
  );

  private readonly appService = inject(AppService);
  private readonly extensionService = inject(ExtensionService);
  private readonly nodeDataProviderExtensionService = inject(
    NodeDataProviderExtensionService
  );
  private readonly dialog = inject(MatDialog);
  private readonly snackBar = inject(MatSnackBar);

  handleClickExtension(extension: NodeDataProviderExtension) {
    // Show run task dialog.
    const data: RunNdpExtensionDialogData = {
      extension,
    };
    const dialogRef = this.dialog.open(RunNdpExtensionDialog, {
      width: '400px',
      data,
    });

    // Process result.
    dialogRef
      .afterClosed()
      .subscribe(async (result?: RunNdpExtensionDialogResult) => {
        if (result == null) {
          return;
        }

        this.onRunDialogRunClicked.emit();

        const modelGraph = this.appService.getModelGraphFromSelectedPane();
        if (modelGraph == null) {
          return;
        }

        // Kick off ndp run.
        const runId = `ndp_run_${ndpId++}`;
        this.nodeDataProviderExtensionService.setRunStatus(
          extension.id,
          runId,
          true
        );
        const { cmdResp, otherError } =
          await this.extensionService.runNdpExtension(
            extension /* extension */,
            modelGraph.modelPath ?? '' /* model path */,
            result.configValues /* config */
          );
        console.log('runNdpExtension result', runId, cmdResp, otherError);

        // Show error if any.
        const error = cmdResp?.error ?? otherError ?? '';
        if (error !== '') {
          console.error(error);
          this.snackBar.open(error, 'Dismiss', {
            duration: 5000,
          });
        }
        // Add ndp if no error.
        else if (cmdResp?.result != null) {
          this.nodeDataProviderExtensionService.addRun(
            runId,
            result.runName,
            extension.id,
            modelGraph,
            { [modelGraph.id]: cmdResp.result }
          );
        }
        this.nodeDataProviderExtensionService.setRunStatus(
          extension.id,
          runId,
          false
        );
      });
  }

  isExtensionRunning(extensionId: string): boolean {
    const status = this.nodeDataProviderExtensionService.runStatus();
    if (status[extensionId] == null) {
      return false;
    }
    return Object.values(status[extensionId]).some((running) => running);
  }
}
