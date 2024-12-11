import { CommonModule } from '@angular/common';
import { ChangeDetectionStrategy, Component, Inject, ChangeDetectorRef } from '@angular/core';
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
import { ModelItemStatus, type ModelItem } from '../../common/types';
import { genUid } from './common/utils';
import { GraphErrorsDialog } from '../graph_error_dialog/graph_error_dialog';
import { LoggingDialog } from '../logging_dialog/logging_dialog';
import { NodeDataProviderExtensionService } from './node_data_provider_extension_service';
import type { Pane } from './common/types';
import type { LoggingServiceInterface } from '../../common/logging_service_interface';

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
  isProcessingUploadRequest = false;

  executionProgress = 0;
  executionTotal = 0;

  constructor(
    @Inject('LoggingService')
    private readonly loggingService: LoggingServiceInterface,
    @Inject('ModelLoaderService')
    private readonly modelLoaderService: ModelLoaderServiceInterface,
    private readonly nodeDataProviderExtensionService: NodeDataProviderExtensionService,
    private readonly appService: AppService,
    private readonly urlService: UrlService,
    private readonly dialog: MatDialog,
    private readonly snackBar: MatSnackBar,
    private changeDetectorRef: ChangeDetectorRef
  ) {}

  ngOnInit() {
    if (this.modelLoaderService.selectedOptimizationPolicy() === '') {
      this.modelLoaderService.selectedOptimizationPolicy.update(() => {
        const curExtensionId = this.getCurrentGraphInformation().models[0].selectedAdapter?.id ?? '';

        return this.modelLoaderService.getOptimizationPolicies(curExtensionId)[0] || '';
      });
    }
  }

  private poolForStatusUpdate(modelItem: ModelItem, modelPath: string, updateCallback: (progress: number, total: number, stdout?: string) => void | Promise<void>, doneCallback: (status: 'done' | 'timeout') => void | Promise<void>, errorCallback: (error: string) => void | Promise<void>) {
    const POOL_TIME_MS = 500;
    const TIMEOUT_MS = 5 * 60 * 1000;

    const startTime = Date.now();
    const intervalId = setInterval(async () => {
      const { isDone, total = 100, progress, error, stdout } = await this.modelLoaderService.checkExecutionStatus(modelItem, modelPath);

      if (error) {
        errorCallback(error);
        clearInterval(intervalId);
        return;
      }

      if (isDone) {
        doneCallback('done');
        clearInterval(intervalId);
        return;
      }

      const deltaTime = Date.now() - startTime;
      if (deltaTime > TIMEOUT_MS) {
        doneCallback('timeout');
        clearInterval(intervalId);
        return;
      }

      if (progress !== -1) {
        updateCallback(progress, total, stdout);
      }
    }, POOL_TIME_MS);
  }

  private async updateGraphInformation(curModel: ModelItem, models: ModelItem[], curPane?: Pane) {
    const newGraphCollections = await this.modelLoaderService.loadModel(curModel);

    if (curModel.status() !== ModelItemStatus.ERROR) {
      this.modelLoaderService.loadedGraphCollections.update((prevGraphCollections) => {
        const curChanges = this.modelLoaderService.changesToUpload();
        if (Object.keys(curChanges).length > 0) {
          newGraphCollections.forEach((graphCollection) => {
            graphCollection.graphs.forEach((graph) => {
              graph.nodes.forEach((node) => {
                const nodeChanges = curChanges[graphCollection.label][node.id] ?? [];

                nodeChanges.forEach(({ key, value }) => {
                  const nodeToUpdate = node.attrs?.find(({ key: nodeKey }) => nodeKey === key);

                  if (nodeToUpdate) {
                    nodeToUpdate.value = value;
                  }
                });
              });
            });
          });
        }

        const newGraphCollectionsLabels = newGraphCollections?.map(({ label }) => label) ?? [];
        const filteredGraphCollections = (prevGraphCollections ?? [])?.filter(({ label }) => !newGraphCollectionsLabels.includes(label));
        const mergedGraphCollections = [...filteredGraphCollections, ...newGraphCollections];

        return mergedGraphCollections;
      });

      this.urlService.setUiState(undefined);
      this.urlService.setModels(models?.map(({ path, selectedAdapter }) => {
        return {
          url: path,
          adapterId: selectedAdapter?.id
        };
      }) ?? []);

      this.modelLoaderService.graphErrors.update(() => undefined);

      newGraphCollections.forEach(({ perf_data }) => {
        if (perf_data) {
          const runId = genUid();

          if (curPane?.modelGraph) {
            this.nodeDataProviderExtensionService.addRun(
              runId,
              `${curPane.modelGraph.id} (Performance Trace)`,
              curModel.selectedAdapter?.id ?? '',
              curPane.modelGraph,
              perf_data,
            );
          }
        }
      });

      this.showSuccessMessage('Model updated');
    } else {
      this.showErrorDialog('Graph Execution Error', curModel.errorMessage ?? 'An error has occured');
    }
  }

  private getCurrentGraphInformation() {
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
      width: 'clamp(10rem, 60vw, 60rem)',
      height: 'clamp(10rem, 60vh, 60rem)',
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
      try {
        this.isProcessingExecuteRequest = true;
        this.loggingService.info('Start executing model', curModel.path);

        const result = await this.modelLoaderService.executeModel(curModel);

        if (curModel.status() !== ModelItemStatus.ERROR) {
          if (result) {
            const updateStatus = (progress: number, total: number, stdout?: string) => {
              this.executionProgress = progress ?? this.executionProgress;
              this.executionTotal = total;
              this.loggingService.debug(`Execution progress: ${progress} of ${total}`, curModel.path);

              if (stdout) {
                this.loggingService.info(stdout);
              }

              this.changeDetectorRef.detectChanges();
            };

            const finishUpdate = async (status: 'done' | 'timeout') => {
              if (status === 'timeout') {
                this.loggingService.error('Model execute timeout', curModel.path);
              } else {
                this.loggingService.info('Model execute finished', curModel.path);
                await this.updateGraphInformation(curModel, models, curPane);
                this.loggingService.info('Model updated', curModel.path);
              }

              this.isProcessingExecuteRequest = false;
            };

            const showError = (error: string) => {
              this.executionProgress = 0;
              this.isProcessingExecuteRequest = false;
              this.loggingService.error('Graph Execution Error', error);
              this.showErrorDialog('Graph Execution Error', error);
            };

            this.poolForStatusUpdate(curModel, curModel.path, updateStatus, finishUpdate, showError);
          } else {
            throw new Error("Graph execution resulted in an error");
          }
        } else {
          throw new Error(curModel.errorMessage ?? 'An error has occured');
        }
      } catch (err) {
        const errorMessage = (err as Error).message ?? 'An error has occured';

        this.loggingService.error('Graph Execution Error', errorMessage);
        this.showErrorDialog('Graph Execution Error', errorMessage);
        this.isProcessingExecuteRequest = false;
      }
    }
  }

  async handleClickUploadGraph() {
    const { curModel, curCollection, changesToUpload, models, curPane } = this.getCurrentGraphInformation();

    if (curModel && curCollection && changesToUpload) {
      try {
        this.isProcessingUploadRequest = true;
        this.loggingService.info('Start uploading model', curModel.path);

        const isUploadSuccessful = await this.modelLoaderService.overrideModel(
          curModel,
          curCollection,
          changesToUpload
        );

        this.loggingService.info('Upload finished', curModel.path);

        if (curModel.status() !== ModelItemStatus.ERROR) {
          this.loggingService.info('Updating existing models', curModel.path);

          if (isUploadSuccessful) {
            await this.updateGraphInformation(curModel, models, curPane);

            this.urlService.setUiState(undefined);
            this.urlService.setModels(models?.map(({ path, selectedAdapter }) => {
              return {
                url: path,
                adapterId: selectedAdapter?.id
              };
            }) ?? []);

            this.modelLoaderService.changesToUpload.update(() => ({}));
            this.modelLoaderService.graphErrors.update(() => undefined);

            this.showSuccessMessage('Model uploaded');
          } else {
            throw new Error("Graph upload didn't return any results");
          }
        } else {
          throw new Error(curModel.errorMessage ?? 'An error has occured');
        }
      } catch (err) {
        const errorMessage =  (err as Error)?.message ?? 'An error has occured.';

        this.loggingService.error('Graph Loading Error', errorMessage);
        this.showErrorDialog('Graph Loading Error', errorMessage);
      } finally {
        this.isProcessingUploadRequest = false;
      }
    }
  }

  handleLogDialogOpen() {
    this.dialog.open(LoggingDialog, {
      width: 'clamp(10rem, 80vw, 100rem)',
      height: 'clamp(10rem, 80vh, 100rem)'
    });
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

  get optimizationPolicies(): string[] {
    const curExtensionId = this.getCurrentGraphInformation().models[0].selectedAdapter?.id ?? '';
    return this.modelLoaderService.getOptimizationPolicies(curExtensionId);
  }
}
