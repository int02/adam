import { useState, useEffect } from 'react'

function DataTraining() {
  const [activeTab, setActiveTab] = useState('datasets')
  const [datasets, setDatasets] = useState([])
  const [trainingJobs, setTrainingJobs] = useState([])
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(false)

  // Data ingestion state
  const [ingestFile, setIngestFile] = useState(null)
  const [ingestFormat, setIngestFormat] = useState('json')
  const [ingestName, setIngestName] = useState('')

  // Training state
  const [selectedDatasets, setSelectedDatasets] = useState([])
  const [modelType, setModelType] = useState('attacker')
  const [epochs, setEpochs] = useState(3)
  const [learningRate, setLearningRate] = useState(2e-5)
  const [useADAMTraining, setUseADAMTraining] = useState(true)
  const [adamModelChoice, setAdamModelChoice] = useState('phi-2')
  const [adamOutputName, setAdamOutputName] = useState('adam_jailbreak')

  useEffect(() => {
    loadDatasets()
    loadModels()
  }, [])

  const loadDatasets = async () => {
    try {
      const response = await fetch('/api/data/datasets')
      const data = await response.json()
      if (data.status === 'success') {
        setDatasets(data.datasets)
      }
    } catch (error) {
      console.error('Error loading datasets:', error)
    }
  }

  const loadModels = async () => {
    try {
      const response = await fetch('/api/models')
      const data = await response.json()
      if (data.status === 'success') {
        setModels(data.models)
      }
    } catch (error) {
      console.error('Error loading models:', error)
    }
  }

  const handleFileUpload = async () => {
    if (!ingestFile) return

    setLoading(true)
    try {
      const formData = new FormData()
      formData.append('file', ingestFile)
      formData.append('format_type', ingestFormat)
      formData.append('name', ingestName || ingestFile.name.split('.')[0])

      const response = await fetch('/api/data/upload-external', {
        method: 'POST',
        body: formData
      })

      const result = await response.json()
      if (result.status === 'success') {
        alert('Data ingested successfully!')
        loadDatasets()
        setIngestFile(null)
        setIngestName('')
      } else {
        alert('Error: ' + result.error)
      }
    } catch (error) {
      alert('Error uploading file: ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  const collectInternalData = async (dataType) => {
    setLoading(true)
    try {
      const response = await fetch('/api/data/collect-internal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data_type: dataType })
      })

      const result = await response.json()
      if (result.status === 'success') {
        alert('Internal data collected successfully!')
        loadDatasets()
      } else {
        alert('Error: ' + result.error)
      }
    } catch (error) {
      alert('Error collecting data: ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  const cleanDataset = async (datasetPath) => {
    setLoading(true)
    try {
      const response = await fetch('/api/data/clean', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data_path: datasetPath,
          cleaning_steps: ['remove_duplicates', 'filter_quality', 'normalize_text']
        })
      })

      const result = await response.json()
      if (result.status === 'success') {
        alert(`Data cleaned! ${result.original_count} → ${result.cleaned_count} samples`)
        loadDatasets()
      } else {
        alert('Error: ' + result.error)
      }
    } catch (error) {
      alert('Error cleaning data: ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  const startTraining = async () => {
    if (selectedDatasets.length === 0 && !useADAMTraining) {
      alert('Please select at least one dataset')
      return
    }

    setLoading(true)
    try {
      if (useADAMTraining) {
        // Use ADAM efficient training
        const response = await fetch('/api/train_adam', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model_name: adamModelChoice,
            output_name: adamOutputName,
            epochs: epochs
          })
        })

        const result = await response.json()
        if (result.status === 'success') {
          alert(`ADAM training started successfully!\nModel will be saved as: ${adamOutputName}\nEstimated cost: ${result.training_info?.cost_estimate || 'Low'}`)
          loadModels()
        } else {
          alert('ADAM training error: ' + result.error)
        }
      } else {
        // Use regular training
        const response = await fetch('/api/train', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            dataset_paths: selectedDatasets,
            model_type: modelType,
            epochs: epochs,
            learning_rate: learningRate
          })
        })

        const result = await response.json()
        if (result.status === 'success') {
          alert('Training started successfully!')
          loadModels()
        } else {
          alert('Training error: ' + result.error)
        }
      }
    } catch (error) {
      alert('Error starting training: ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  const tabs = [
    { id: 'datasets', label: 'Datasets' },
    { id: 'training', label: 'Training' },
    { id: 'models', label: 'Models' }
  ]

  return (
    <div className="data-training">
      <h2>Data Management & Model Training</h2>

      <div className="tabs">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="tab-content">
        {activeTab === 'datasets' && (
          <div className="datasets-tab">
            <div className="section">
              <h3>Ingest External Data</h3>
              <div className="form-group">
                <input
                  type="file"
                  onChange={(e) => setIngestFile(e.target.files[0])}
                  accept=".json,.csv,.txt"
                />
                <select
                  value={ingestFormat}
                  onChange={(e) => setIngestFormat(e.target.value)}
                >
                  <option value="json">JSON</option>
                  <option value="csv">CSV</option>
                  <option value="txt">Text</option>
                </select>
                <input
                  type="text"
                  placeholder="Dataset name (optional)"
                  value={ingestName}
                  onChange={(e) => setIngestName(e.target.value)}
                />
                <button
                  onClick={handleFileUpload}
                  disabled={!ingestFile || loading}
                >
                  {loading ? 'Uploading...' : 'Upload & Ingest'}
                </button>
              </div>
            </div>

            <div className="section">
              <h3>Collect Internal Data</h3>
              <div className="button-group">
                <button onClick={() => collectInternalData('vulnerabilities')} disabled={loading}>
                  Collect Vulnerabilities
                </button>
                <button onClick={() => collectInternalData('logs')} disabled={loading}>
                  Collect Logs
                </button>
              </div>
            </div>

            <div className="section">
              <h3>Available Datasets</h3>
              <div className="datasets-list">
                {datasets.map((dataset, index) => (
                  <div key={index} className="dataset-item">
                    <div className="dataset-info">
                      <span className="dataset-name">{dataset.name}</span>
                      <span className="dataset-type">{dataset.type}</span>
                      <span className="dataset-count">{dataset.count} samples</span>
                      <span className="dataset-size">{dataset.size_mb.toFixed(2)} MB</span>
                    </div>
                    <div className="dataset-actions">
                      <button onClick={() => cleanDataset(dataset.path)} disabled={loading}>
                        Clean Data
                      </button>
                    </div>
                  </div>
                ))}
                {datasets.length === 0 && (
                  <p>No datasets available. Upload external data or collect internal data first.</p>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'training' && (
          <div className="training-tab">
            <div className="section">
              <h3>Start Model Training</h3>
              <div className="training-form">
                <div className="form-group">
                  <label>
                    <input
                      type="checkbox"
                      checked={useADAMTraining}
                      onChange={(e) => setUseADAMTraining(e.target.checked)}
                    />
                    Use ADAM Cost-Effective Training (Recommended)
                  </label>
                </div>

                {useADAMTraining ? (
                  // ADAM Training Options
                  <div className="adam-training-options">
                    <div className="form-row">
                      <div className="form-group">
                        <label>ADAM Model:</label>
                        <select value={adamModelChoice} onChange={(e) => setAdamModelChoice(e.target.value)}>
                          <option value="phi-2">Phi-2 (2.7B - Best Balance)</option>
                          <option value="gemma-2b">Gemma 2B (Small & Fast)</option>
                          <option value="qwen-1.5b">Qwen 1.5B (Very Efficient)</option>
                          <option value="qwen-0.5b">Qwen 0.5B (Minimal)</option>
                          <option value="mistral-7b">Mistral 7B (More Capable)</option>
                        </select>
                      </div>

                      <div className="form-group">
                        <label>Output Name:</label>
                        <input
                          type="text"
                          value={adamOutputName}
                          onChange={(e) => setAdamOutputName(e.target.value)}
                          placeholder="adam_jailbreak"
                        />
                      </div>
                    </div>

                    <div className="adam-info">
                      <div className="info-item">
                        <strong>Cost-Effective:</strong> Uses LoRA fine-tuning (only 0.1-1% of parameters)
                      </div>
                      <div className="info-item">
                        <strong>Memory Efficient:</strong> 4-bit quantization
                      </div>
                      <div className="info-item">
                        <strong>Quick Training:</strong> 1-3 epochs typically sufficient
                      </div>
                    </div>
                  </div>
                ) : (
                  // Regular Training Options
                  <div className="form-group">
                    <label>Select Datasets:</label>
                    <div className="dataset-selection">
                      {datasets.map((dataset, index) => (
                        <label key={index} className="checkbox-label">
                          <input
                            type="checkbox"
                            checked={selectedDatasets.includes(dataset.path)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setSelectedDatasets([...selectedDatasets, dataset.path])
                              } else {
                                setSelectedDatasets(selectedDatasets.filter(p => p !== dataset.path))
                              }
                            }}
                          />
                          {dataset.name} ({dataset.count} samples)
                        </label>
                      ))}
                    </div>
                  </div>
                )}

                {!useADAMTraining && (
                  <div className="form-row">
                    <div className="form-group">
                      <label>Model Type:</label>
                      <select value={modelType} onChange={(e) => setModelType(e.target.value)}>
                        <option value="attacker">Attacker Model</option>
                        <option value="evaluator">Evaluator Model</option>
                        <option value="feedback">Feedback Model</option>
                        <option value="refiner">Refiner Model</option>
                      </select>
                    </div>

                    <div className="form-group">
                      <label>Learning Rate:</label>
                      <input
                        type="number"
                        value={learningRate}
                        onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                        step="1e-6"
                        min="1e-6"
                        max="1e-3"
                      />
                    </div>
                  </div>
                )}

                <div className="form-row">
                  <div className="form-group">
                    <label>Epochs:</label>
                    <input
                      type="number"
                      value={epochs}
                      onChange={(e) => setEpochs(parseInt(e.target.value))}
                      min="1"
                      max={useADAMTraining ? "5" : "10"}
                    />
                    <small>{useADAMTraining ? "1-3 epochs usually sufficient" : "Full training may take longer"}</small>
                  </div>
                </div>

                <button
                  onClick={startTraining}
                  disabled={selectedDatasets.length === 0 || loading}
                  className="train-button"
                >
                  {loading ? 'Training...' : 'Start Training'}
                </button>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'models' && (
          <div className="models-tab">
            <div className="section">
              <h3>Trained Models</h3>
              <div className="models-list">
                {models.map((model, index) => (
                  <div key={index} className="model-item">
                    <div className="model-info">
                      <span className="model-name">{model.name}</span>
                      <span className="model-path">{model.path}</span>
                      <span className="model-date">
                        {new Date(model.created * 1000).toLocaleString()}
                      </span>
                    </div>
                    <div className="model-actions">
                      <button>Use This Model</button>
                    </div>
                  </div>
                ))}
                {models.length === 0 && (
                  <p>No trained models available. Train a model first.</p>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default DataTraining