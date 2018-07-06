require 'torch'
require 'nn'
require 'optim'
require 'cunn'
require 'cudnn'
require 'gnuplot'
dofile('../Stream_image_classification/sup_functions.lua')
opt = {
   dataset = 'lsun',       -- imagenet / lsun / folder
   batch_size = 100,
   nz = 100,               -- #  of dim for Z
   nThreads = 4,           -- #  of data loading threads to use
   niter = 10,             -- #  of iter at starting learning rate
   lr = 0.0003,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'experiment1',
   noise = 'normal',       -- uniform / normal
   nb_classes = 10
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end
dp = 0
local function initialize_autoencoder(feature_size)
  local encoder = nn.Sequential()
  local decoder = nn.Sequential()
--  local activation1 = nn.ReLU()
  -- TRY DROPOUT
  --local enc_arch = {feature_size, 1024, 512, 256, 128, 64, 32, 16, 8, 2}
  --local dec_arch = {2, 8, 16, 32, 64, 128, 256, 512, 1024, feature_size}
  local enc_arch = {feature_size, 512, 128, 32, 16}
  local dec_arch = {16, 8, 32, 128, 512, feature_size}
  for layer = 1, #enc_arch-2 do
    encoder:add(nn.Linear(enc_arch[layer], enc_arch[layer+1])):add(nn.ReLU(true)):add(nn.Dropout(dp))
    --encoder:add(nn.BatchNormalization(enc_arch[layer+1]))
  end
  encoder:add(nn.Linear(enc_arch[#enc_arch-1], enc_arch[#enc_arch]))
  for layer = 1, #dec_arch-2 do
    decoder:add(nn.Linear(dec_arch[layer], dec_arch[layer+1])):add(nn.ReLU(true)):add(nn.Dropout(dp))
    --decoder:add(nn.BatchNormalization(dec_arch[layer+1]))
  end
  decoder:add(nn.Linear(dec_arch[#dec_arch-1], dec_arch[#dec_arch]))
  -----------------------------------------------------------------
  --model:apply(weights_init)
  return encoder, decoder
end  

function get_batch(data, batch_size)
  local indices = torch.randperm(data:size(1))[{{1, batch_size}}]
  return data:index(1, indices:long())
end
-- local function test_classifier(classifier, data, current_class)
--   local  res_ = classifier:forward(data:cuda())
--   _, ids_ = torch.max(res_,2); local labels_ = ids_:eq(current_class)
--   return labels_:sum()/1000
-- end

classifier = torch.load('../Stream_image_classification/models/classifiers/LSUN_real_data_classifier.t7')
feature_size = 2048
data = torch.load('../Stream_image_classification/subsets/LSUN/100k_images_10_classes/full_data_orig.t7')
testset = torch.load('/home/besedin/workspace/Data/LSUN/data_t7/validation/extracted_features/full_dataset_10_classes.t7')
confusion = test_classifier(classifier, testset)
print(confusion)
--orig_data = data.data:cuda()
encoder, decoder = initialize_autoencoder(feature_size)
full_model = nn.Sequential()
full_model:add(encoder):add(decoder)
criterion = nn.MSECriterion()
---------------------------------------------------------------------------
optimState = {
   learningRate = opt.lr,
   learningRateDecay = 1e-7,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local input = torch.Tensor(opt.batch_size, feature_size)
local errD, errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input = input:cuda()
   if pcall(require, 'cudnn') then
      require 'cudnn'
      cudnn.benchmark = true
      cudnn.convert(full_model, cudnn)
   end
   full_model:cuda(); criterion:cuda()
end

local parameters, gradParameters = full_model:getParameters()

if opt.display then disp = require 'display' end



local fx = function(x)
   gradParameters:zero()
   local real = get_batch(data.data, opt.batch_size)
   input:copy(real)
   local output = full_model:forward(input)
   err_real = criterion:forward(output, input)
   local df_do = criterion:backward(output, input)
   full_model:backward(input, df_do)
   --print(err_real)
   return err_real, gradParameters
end

-- train
--res_real = classifier:forward(orig_data)
--_, ids = torch.max(res_real,2); labels_real = ids:eq(current_class)
--print('Correctly classified from real data: ' .. labels_real:sum()/1000 .. '%')
local function display_images()
  local real = get_batch(data.data, opt.batch_size)
  local fake = full_model:forward(real:cuda())
  disp.image(fake, {win=opt.display_id, title=opt.name})
  disp.image(real, {win=opt.display_id * 3, title=opt.name})
end

local function visualize_features(data, encoder)
  local to_plot = {}
  for idx = 1, 10 do
    features = encoder:forward(data.data[{{1 + (idx-1)*100000, 1000 + (idx-1)*100000},{}}]:cuda()):float()
    to_plot[idx] = {features[{{},{1}}]:squeeze(), features[{{},{2}}]:squeeze(), 'with points'}
  end
  gnuplot.plot(to_plot)
end
gnuplot.figure()
reconstructed_data = {}
noise_ = torch.CudaTensor(parameters:size(1))
for epoch = 1, opt.niter do
  local counter = 0
  full_model:training()
  for i = 1, math.min(data.data:size(1), opt.ntrain), opt.batch_size do
    xlua.progress(i, data.data:size(1))
    tm:reset()
    -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    optim.adam(fx, parameters, optimState);
    --parameters:add(noise_:normal(0, 0.001))
    -- display
    counter = counter + 1
    if counter % 100 == 0 and opt.display then
      full_model:evaluate()
      display_images()
      full_model:training()
    end
    
    if (i-1)%10000==0 then
      full_model:evaluate()
      features = encoder:forward(testset.data:cuda())
      reconstructed_data.data = decoder:forward(features)
      reconstructed_data.labels = testset.labels
      confusion_reconstructed = test_classifier(classifier, reconstructed_data)
      print(confusion_reconstructed)
      visualize_features(data, encoder)
      full_model:training()
    end
  end
  parameters, gradParameters = nil, nil -- nil them to avoid spiking memory
  parameters, gradParameters = full_model:getParameters() -- reflatten the params and get them
  print('End of epoch')
  full_model:evaluate()
  -- TESTING
  features = encoder:forward(testset.data:cuda())
  reconstructed_data.data = decoder:forward(features)
  reconstructed_data.labels = testset.labels
  confusion_reconstructed = test_classifier(classifier, reconstructed_data)
  print(confusion_reconstructed)
--  print('Correctly classified from reconstructed data: ' .. res .. '%')
end

res = {}
res.features = torch.FloatTensor(1e+6, 2)
res.labels = data.labels
reconstructed_data = torch.FloatTensor(1e+6, 2048)
for idx = 1, 10000 do
  xlua.progress(idx, 10000)
  res.features[{{1 + (idx-1)*100, idx*100},{}}] = encoder:forward(data.data[{{1 + (idx-1)*100, idx*100},{}}]:cuda()):float()
  reconstructed_data[{{1 + (idx-1)*100, idx*100},{}}] = decoder:forward(res.features[{{1 + (idx-1)*100, idx*100},{}}]:cuda()):float()
end

decoder = decoder:clearState()
encoder = encoder:clearState()

torch.save('./encoded_data.t7', res)
torch.save('./pretrained_decoder.t7', decoder)
