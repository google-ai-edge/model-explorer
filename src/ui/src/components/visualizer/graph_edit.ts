import { CommonModule } from '@angular/common';
import { ChangeDetectionStrategy, Component, Inject } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
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
  isProcessingExecuteRequest = false;

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

  private showErrorDialog(title: string, ...messages: string[]) {
    this.modelLoaderService.graphErrors.update((curErrors) => {
      return [...new Set([...curErrors ?? [], ...messages])];
    });
    this.dialog.open(GraphErrorsDialog, {
      width: 'clamp(10rem, 30vmin, 30rem)',
      height: 'clamp(10rem, 30vmin, 30rem)',
      data: {
        errorMessages: [...messages],
        title
      }
    });
  }

  private showSuccessMessage(message: string, action = 'Dismiss') {
    this.snackBar.open(message, action, {
      duration: 5000,
      verticalPosition: 'top',
      horizontalPosition: 'center'
    });
  }

  async handleClickExecuteGraph() {
    const { curModel, curPane, models } = this.getCurrentGraphInformation();

    if (curModel) {
      this.isProcessingExecuteRequest = true;

      const result = await this.modelLoaderService.executeModel(curModel);

      if (curModel.status() !== ModelItemStatus.ERROR) {
        if (result) {
          const newGraphCollections = await this.modelLoaderService.loadModel(curModel);

          if (curModel.status() !== ModelItemStatus.ERROR) {
            this.modelLoaderService.loadedGraphCollections.update((prevGraphCollections) => {
              const curChanges = this.modelLoaderService.changesToUpload();
              if (Object.keys(curChanges).length > 0) {
                newGraphCollections.forEach((graphCollection) => {
                  const graphChanges = curChanges[graphCollection.label];
                  if (graphChanges) {
                    graphCollection.graphs.forEach((graph) => {
                      graph.nodes.forEach((node) => {
                        const nodeChanges = graphChanges[node.id];

                        if (nodeChanges && nodeChanges.length > 0) {
                          // TODO: apply changes
                        }
                      });
                    });
                  }
                });
                console.log(result);
                debugger;
              }

              const newGraphCollectionsLabels = newGraphCollections.map(({ label }) => label);
              const filteredGraphCollections = (prevGraphCollections ?? [])?.filter(({ label }) => !newGraphCollectionsLabels.includes(label));
              const mergedGraphCollections = [...filteredGraphCollections, ...newGraphCollections];

              return mergedGraphCollections;
            });

            this.urlService.setUiState(undefined);
            this.urlService.setModels(models.map(({ path, selectedAdapter }) => {
              return {
                url: path,
                adapterId: selectedAdapter?.id
              };
            }));

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

            this.showSuccessMessage('Model updated');
          } else {
            this.showErrorDialog('Graph Execution Error', curModel.errorMessage ?? 'An error has occured');
          }
        } else {
          this.showErrorDialog('Graph Execution Error', "Graph execution didn't return any results");
        }
      } else {
        this.showErrorDialog('Graph Execution Error', curModel.errorMessage ?? 'An error has occured');
      }

      this.isProcessingExecuteRequest = false;
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

          this.showSuccessMessage('Model uploaded');
        } else {
          this.showErrorDialog('Graph Loading Error', "Graph upload didn't return any results");
        }
      } else {
        this.showErrorDialog('Graph Loading Error', curModel.errorMessage ?? 'An error has occured');
      }
    }
  }

  handleClickSelectOptimizationPolicy(evt: Event) {
    const optimizationPolicy = (evt.target as HTMLSelectElement).value;
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
