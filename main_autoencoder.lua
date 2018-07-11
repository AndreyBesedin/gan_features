require 'torch'
require 'nn'
require 'optim'
require 'cunn'
require 'cudnn'

opt = {
   dataset = 'lsun',       -- imagenet / lsun / folder
   batch_size = 100,
   nz = 100,               -- #  of dim for Z
   nThreads = 4,           -- #  of data loading threads to use
   niter = 5,             -- #  of iter at starting learning rate
   lr = 0.0001,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'experiment1',
   noise = 'normal',       -- uniform / normal
}

data_classes = {'bedroom', 'bridge', 'church_outdoor', 'classroom', 'conference_room',
           'dining_room', 'kitchen', 'living_room', 'restaurant', 'tower'}
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
function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

function initialize_autoencoder(feature_size)
  local encoder = nn.Sequential()
  local decoder = nn.Sequential()
  -- TRY DROPOUT
  encoder:add(nn.Linear(feature_size, 512)):add(nn.LeakyReLU(0.2, true))
  encoder:add(nn.BatchNormalization(512))
  encoder:add(nn.Linear(512, 128)):add(nn.LeakyReLU(0.2, true))
  encoder:add(nn.BatchNormalization(128))
  encoder:add(nn.Linear(128, 32)):add(nn.LeakyReLU(0.2, true))
  encoder:add(nn.BatchNormalization(32))
  encoder:add(nn.Linear(32, 2))
  ----------------------------------------------------------------
  decoder:add(nn.Linear(2, 64)):add(nn.LeakyReLU(0.2, true))
  decoder:add(nn.BatchNormalization(64))
  decoder:add(nn.Linear(64, 256)):add(nn.LeakyReLU(0.2, true))
  decoder:add(nn.BatchNormalization(256))
  decoder:add(nn.Linear(256, feature_size)) 
  -----------------------------------------------------------------
  --model:apply(weights_init)
  return encoder, decoder
end  

function get_batch(data, batch_size)
  local indices = torch.randperm(data:size(1))[{{1, batch_size}}]
  return data:index(1, indices:long())
end
function test_classifier(classifier, data, current_class)
  local  res_ = classifier:forward(data:cuda())
  _, ids_ = torch.max(res_,2); local labels_ = ids_:eq(current_class)
  return labels_:sum()/1000
end

function magic_criterion_1(C, C_star, real_label)
  -- Cost function is a L2 norm comparing the classifier output on original and reconstructed data
  -- For L2 norm gradient value depends on the distance between the outputs and represents both direction and magnitude of proposed changes
  local dErr_dCstar = torch.add(torch.cmul(real_label, nn.ReLU():forward(torch.add(C, - C_star))), -torch.cmul(1 - real_label, nn.ReLU():forward(torch.add(C_star, - C))))
  return dErr_dCstar
end

function magic_criterion_2(C, C_star, real_label)
  -- Cost function is a L1 norm comparing the classifier output on original and reconstructed data
  -- For L1 norm gradient value is independent from the distance between the output and only represents the direction with magnitude of 1
  local dErr_dCstar = torch.add(torch.cmul(real_label, torch.add(C, - C_star):gt(0):float()), -torch.cmul(1 - real_label, torch.add(C_star, - C):gt(0):float()))
  return dErr_dCstar
end

classifier = torch.load('../Stream_image_classification/models/classifiers/LSUN_real_data_classifier.t7')
training_classifier = classifier:clone()
training_classifier:remove(8)
training_classifier:add(nn.SoftMax())

for idx_class = 1, 1 do
  current_class = idx_class
  data_class = data_classes[current_class]
  feature_size = 2048
  data = torch.load('../Stream_image_classification/subsets/LSUN/100k_images_10_classes/'..data_class..'_real.t7')
  orig_data = data:cuda()
  encoder, decoder = initialize_autoencoder(feature_size)
  full_model = nn.Sequential()
  full_model:add(encoder):add(decoder)
  criterion = nn.AbsCriterion()
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
    -- train with real
    local real = get_batch(data, opt.batch_size)
    input:copy(real)
    local output = full_model:forward(input)
    local C_real = training_classifier:forward(real)
    local C_reconstructed = training_classifier:forward(output)
    err_real = criterion:forward(output, input)
    --print(err_real)
    --full_model:training()
    local df_do = criterion:backward(output, input)
    full_model:backward(input, df_do)
    --full_model:evaluate()
    -- train with fake
    return err_real, gradParameters
  end

  -- train
  res_real = classifier:forward(orig_data)
  _, ids = torch.max(res_real,2); labels_real = ids:eq(current_class)
  print('Correctly classified from real data: ' .. labels_real:sum()/1000 .. '%')
  for epoch = 1, opt.niter do
    epoch_tm:reset()
    local counter = 0
    full_model:training()
    for i = 1, math.min(data:size(1), opt.ntrain), opt.batch_size do
      xlua.progress(i, data:size(1))
      tm:reset()
      -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      optim.adam(fx, parameters, optimState);
      -- display
      counter = counter + 1
      if counter % 10 == 0 and opt.display then
        local real = get_batch(data, opt.batch_size)
        local fake = full_model:forward(real:cuda())
        disp.image(fake, {win=opt.display_id, title=opt.name})
        disp.image(real, {win=opt.display_id * 3, title=opt.name})
      end
    end
    parameters, gradParameters = nil, nil -- nil them to avoid spiking memory
    parameters, gradParameters = full_model:getParameters() -- reflatten the params and get them
    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
    full_model:evaluate()
    features = encoder:forward(data:cuda())
    reconstructed_data = decoder:forward(features)
    res = test_classifier(classifier, reconstructed_data, current_class)
    print('Correctly classified from reconstructed data: ' .. res .. '%')
  end
  features = encoder:forward(data:cuda())
  reconstructed_data = decoder:forward(features)
  torch.save('../Stream_image_classification/subsets/LSUN/100k_images_10_classes/'..data_class..'_features_autoencoder_bn.t7', features)
  torch.save('../Stream_image_classification/subsets/LSUN/100k_images_10_classes/'..data_class..'_reconstructed_bn.t7', reconstructed_data)
  decoder = decoder:clearState()
  encoder = encoder:clearState()
  torch.save('../Stream_image_classification/subsets/LSUN/100k_images_10_classes/'..data_class..'_encoder_bn.t7', encoder)
  torch.save('../Stream_image_classification/subsets/LSUN/100k_images_10_classes/'..data_class..'_decoder_bn.t7', decoder)
end
