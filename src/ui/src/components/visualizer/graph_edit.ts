import { CommonModule } from '@angular/common';
import { ChangeDetectionStrategy, Component, Inject } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatButtonToggleModule } from '@angular/material/button-toggle';
import { MatDialogModule, MatDialog } from '@angular/material/dialog';
import { MatIconModule } from '@angular/material/icon';
import { MatMenuModule } from '@angular/material/menu';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatTooltipModule } from '@angular/material/tooltip';
import type { ModelLoaderServiceInterface } from '../../common/model_loader_service_interface';
import { AppService } from './app_service';
import { UrlService } from '../../services/url_service';
import { MatSnackBar } from '@angular/material/snack-bar';
import { ModelItemStatus } from '../../common/types';
import { genUid } from './common/utils';
import { ModelGraph } from './common/model_graph';
import { GraphErrorsDialog } from '../graph_error_dialog/graph_error_dialog';
import { NodeDataProviderExtensionService } from './node_data_provider_extension_service';

/**
 * The graph edit component.
 *
 * It allows users to upload changes and execute a graph.
 */
@Component({
  standalone: true,
  selector: 'graph-edit',
  imports: [
    CommonModule,
    MatButtonModule,
    MatButtonToggleModule,
    MatDialogModule,
    MatIconModule,
    MatMenuModule,
    MatProgressSpinnerModule,
    MatTooltipModule,
  ],
  templateUrl: './graph_edit.ng.html',
  styleUrls: ['./graph_edit.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class GraphEdit {
  isExecuteEnabled = true;

  constructor(
    @Inject('ModelLoaderService')
    private readonly modelLoaderService: ModelLoaderServiceInterface,
    private readonly nodeDataProviderExtensionService: NodeDataProviderExtensionService,
    private readonly appService: AppService,
    private readonly urlService: UrlService,
    private readonly dialog: MatDialog,
    private readonly snackBar: MatSnackBar,
  ) {}

    getCurrentGraphInformation() {
    const curPane = this.appService.getSelectedPane();
    const curCollectionLabel = curPane?.modelGraph?.collectionLabel;
    const curCollection = this.appService.curGraphCollections().find(({ label }) =>label === curCollectionLabel);
    const models = this.modelLoaderService.models();
    const curModel = models.find(({ label }) => label === curCollectionLabel);
    const changesToUpload = this.modelLoaderService.changesToUpload()[curCollectionLabel ?? ''];

    return {
      curModel,
      curCollection,
      curCollectionLabel,
      curPane,
      models,
      changesToUpload,
    };
  }

  async handleClickExecuteGraph() {
    console.log('clicked');
    const { curModel, curPane, models } = this.getCurrentGraphInformation();

    if (curModel) {
      this.isExecuteEnabled = false;

      const result = await this.modelLoaderService.executeModel(curModel);

      if (curModel.status() !== ModelItemStatus.ERROR) {
        if (result) {
          this.modelLoaderService.loadedGraphCollections.update((prevGraphCollections) => {
            if (!prevGraphCollections) {
              return undefined;
            }

            return [...prevGraphCollections];
          });

          this.urlService.setUiState(undefined);
          this.urlService.setModels(models.map(({ path, selectedAdapter }) => {
            return {
              url: path,
              adapterId: selectedAdapter?.id
            };
          }));

          this.modelLoaderService.changesToUpload.update(() => ({}));
          this.modelLoaderService.graphErrors.update(() => undefined);

          if (result.perf_data) {
            const runId = genUid();
            const modelGraph = curPane?.modelGraph as ModelGraph;

            this.nodeDataProviderExtensionService.addRun(
              runId,
              `${modelGraph.id} (Performance Trace)`,
              curModel.selectedAdapter?.id ?? '',
              modelGraph,
              result.perf_data,
            );
          }

          this.snackBar.open('Model updated', 'Dismiss', {
            duration: 5000,
            verticalPosition: 'top',
            horizontalPosition: 'center'
          });
        } else {
          this.modelLoaderService.graphErrors.update((curErrors) => {
            return [...new Set([...curErrors ?? [], "Graph execution didn't return any results"])];
          });
          this.dialog.open(GraphErrorsDialog, {
            width: 'clamp(10rem, 30vmin, 30rem)',
            height: 'clamp(10rem, 30vmin, 30rem)',
            data: {
              errorMessages: [this.graphHasErrors],
              title: 'Graph Execution Errors'
            }
          });
        }
      } else {
        this.modelLoaderService.graphErrors.update((curErrors) => {
          return [...new Set([...curErrors ?? [], curModel.errorMessage ?? ''])];
        });
        this.dialog.open(GraphErrorsDialog, {
          width: 'clamp(10rem, 30vmin, 30rem)',
          height: 'clamp(10rem, 30vmin, 30rem)',
          data: {
            errorMessages: [curModel.errorMessage ?? ''],
            title: 'Graph Execution Errors'
          }
        });
      }

      this.isExecuteEnabled = true;
    }
  }

  async handleClickUploadGraph() {
    const { curModel, curCollection, curCollectionLabel, changesToUpload, models } = this.getCurrentGraphInformation();

    if (curModel && curCollection && changesToUpload) {
      const updatedGraphCollection = await this.modelLoaderService.overrideModel(
        curModel,
        curCollection,
        changesToUpload
      );

      if (curModel.status() !== ModelItemStatus.ERROR) {
        if (updatedGraphCollection) {
          this.modelLoaderService.loadedGraphCollections.update((prevGraphCollections) => {
            if (!prevGraphCollections) {
              return undefined;
            }

            const collectionToUpdate = prevGraphCollections.findIndex(({ label }) => label === curCollectionLabel) ?? -1;

            if (collectionToUpdate !== -1) {
              prevGraphCollections[collectionToUpdate] = updatedGraphCollection;
            }

            return [...prevGraphCollections];
          });

          this.urlService.setUiState(undefined);
          this.urlService.setModels(models.map(({ path, selectedAdapter }) => {
            return {
              url: path,
              adapterId: selectedAdapter?.id
            };
          }));

          this.modelLoaderService.changesToUpload.update(() => ({}));
          this.modelLoaderService.graphErrors.update(() => undefined);
        }
      } else {
        this.modelLoaderService.graphErrors.update((curErrors) => {
          return [...new Set([...curErrors ?? [], curModel.errorMessage ?? ''])];
        });
        this.dialog.open(GraphErrorsDialog, {
          width: 'clamp(10rem, 30vmin, 30rem)',
          height: 'clamp(10rem, 30vmin, 30rem)',
          data: {
            errorMessages: [curModel.errorMessage ?? ''],
            title: 'Graph Loading Errors'
          }
        });
      }
    }
  }

  handleClickSelectOptimizationPolicy(optimizationPolicy: string) {
    this.modelLoaderService.selectedOptimizationPolicy.update(() => optimizationPolicy);
  }

  get hasChangesToUpload() {
    return this.modelLoaderService.hasChangesToUpload;
  }

  get hasCurModel() {
    return this.getCurrentGraphInformation().curModel !== undefined;
  }

  get graphHasErrors() {
    return this.modelLoaderService.graphErrors() !== undefined;
  }

  get selectedOptimizationPolicy(): string {
    const curExtensionId = this.getCurrentGraphInformation().models[0].selectedAdapter?.id ?? '';
    return this.modelLoaderService.selectedOptimizationPolicy() || (this.modelLoaderService.getOptimizationPolicies(curExtensionId)[0] ?? 'Default');
  }

  get optimizationPolicies(): string[] {
    const curExtensionId = this.getCurrentGraphInformation().models[0].selectedAdapter?.id ?? '';
    return this.modelLoaderService.getOptimizationPolicies(curExtensionId);
  }
}
