require 'torch'
require 'nn'
require 'optim'
dofile('./sup_functions.lua')
opt = {
   dataset = 'lsun',       -- imagenet / lsun / folder
   batch_size = 128,
   nz = 100,               -- #  of dim for Z
   nThreads = 4,           -- #  of data loading threads to use
   niter = 20,             -- #  of iter at starting learning rate
   lr = 0.0001,            -- initial learning rate for adam
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

local nc = 3
local nz = opt.nz
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

local function initialize_generator(noise_size, feature_size)
  local generator = nn.Sequential()
  generator:add(nn.Linear(noise_size, 256)):add(nn.ReLU(true))
--  generator:add(nn.BatchNormalization(256))
  generator:add(nn.Linear(256, 512)):add(nn.ReLU(true))
--  generator:add(nn.BatchNormalization(512))
  generator:add(nn.Linear(512, feature_size))
--  generator:add(nn.Tanh(true))
  generator = require('weight-init')(generator, 'heuristic')
  --generator:apply(weights_init)
  return generator
end

local function initialize_discriminator(feature_size)
  local discriminator = nn.Sequential()
  discriminator:add(nn.Linear(feature_size, feature_size + 128)):add(nn.ReLU(true))
--  discriminator:add(nn.BatchNormalization(feature_size + 128))
  discriminator:add(nn.Linear(feature_size + 128, 512)):add(nn.ReLU(true))  
--  discriminator:add(nn.BatchNormalization(512))
  discriminator:add(nn.Linear(512, 32)):add(nn.ReLU(true))
--  discriminator:add(nn.BatchNormalization(32))
  discriminator:add(nn.Linear(32, 4)):add(nn.ReLU(true))
  discriminator:add(nn.Linear(4, 1))
--  discriminator:add(nn.Sigmoid())
  discriminator:add(nn.View(1))
  discriminator = require('weight-init')(discriminator, 'heuristic')
  --discriminator:apply(weights_init)
  return discriminator
end  
noise_size = 100
feature_size = 2048
netG = initialize_generator(noise_size, feature_size)
netD = initialize_discriminator(feature_size)
local criterion_D = nn.MSECriterion()
--local criterion_D = nn.BCECriterion()
local criterion_G = nn.AbsCriterion()
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local input = torch.Tensor(opt.batch_size, feature_size)
local noise = torch.Tensor(opt.batch_size, nz)
local label = torch.Tensor(opt.batch_size)
local errD, errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input = input:cuda();  noise = noise:cuda();  label = label:cuda()
   if pcall(require, 'cudnn') then
      require 'cudnn'
      cudnn.benchmark = true
      cudnn.convert(netG, cudnn)
      cudnn.convert(netD, cudnn)
   end
   netD:cuda();           netG:cuda();           criterion_D:cuda();        criterion_G:cuda()
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.display then disp = require 'display' end

noise_vis = noise:clone()
if opt.noise == 'uniform' then
    noise_vis:uniform(-1, 1)
elseif opt.noise == 'normal' then
    noise_vis:normal(0, 1)
end
data = torch.load('./bedroom_real.t7')
classifier = torch.load('./LSUN_real_data_classifier.t7')
original_data_mean_sample = data:mean(1)
original_data_std_sample = data:std(1)
-- create closure to evaluate f(X) and df/dX of discriminator
function get_batch(data, batch_size)
  local indices = torch.randperm(data:size(1))[{{1, batch_size}}] 
  return data:index(1, indices:long())
end

local fDx = function(x)
   gradParametersD:zero()
   -- train with real
   data_tm:reset(); data_tm:resume()
   local real = get_batch(data, opt.batch_size)
   data_tm:stop()
   input:copy(real)
   label:fill(real_label)
   local output = netD:forward(input)
   local errD_real = criterion_D:forward(output, label)
   local df_do = criterion_D:backward(output, label)
   netD:backward(input, df_do)
   -- train with fake
   if opt.noise == 'uniform' then -- regenerate random noise
       noise:uniform(-1, 1)
   elseif opt.noise == 'normal' then
       noise:normal(0, 1)
   end
   local fake = netG:forward(noise)
   input:copy(fake)
   label:fill(fake_label)
   local output = netD:forward(input)
   local errD_fake = criterion_D:forward(output, label)
   local df_do = criterion_D:backward(output, label)
   netD:backward(input, df_do)
   errD = errD_real + errD_fake
   return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   gradParametersG:zero()
   --[[ the three lines below were already executed in fDx, so save computation
   noise:uniform(-1, 1) -- regenerate random noise
   local fake = netG:forward(noise)
   input:copy(fake) ]]--
   label:fill(real_label) -- fake labels are real for generator cost
   local output = netD.output -- netD:forward(input) was already executed in fDx, so save computation
   errG = criterion_D:forward(output, label)
   local df_do = criterion_D:backward(output, label)
   local df_dg = netD:updateGradInput(input, df_do)
   netG:backward(noise, df_dg)
   return errG, gradParametersG
end

-- train
fake = netG:forward(noise_vis)
testset = {}; testset.data = fake; testset.labels = torch.ones(fake:size(1))
confusion_reconstructed = test_classifier(classifier, testset)
print(confusion_reconstructed)
for epoch = 1, opt.niter do
   epoch_tm:reset()
   local counter = 0
   for i = 1, math.min(data:size(1), opt.ntrain), opt.batch_size do
      tm:reset()
      -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      optim.adam(fDx, parametersD, optimStateD);

      -- (2) Update G network: maximize log(D(G(z)))
      optim.adam(fGx, parametersG, optimStateG);

      -- display
      counter = counter + 1
      if counter % 10 == 0 and opt.display then
         local fake = netG:forward(noise_vis)
         local real = get_batch(data, opt.batch_size)
         disp.image(fake, {win=opt.display_id, title=opt.name})
         disp.image(real, {win=opt.display_id * 3, title=opt.name})
      end

      -- logging
      if ((i-1) / opt.batch_size) % 1 == 0 then
--        netG:evaluate()
	      fake = netG:forward(noise_vis)
	      local distance_fake_to_real = fake:mean(1):float()-original_data_mean_sample
        mean_error = distance_fake_to_real:norm()
	      local std_variation = fake:std(1):float() - original_data_std_sample
	      std_error = std_variation:norm()
	      print('Error of generated samples samples: ' .. mean_error .. '\nElement wise variation difference: ' .. std_error)
--        netG:training()
      end
      if (i-1)/opt.batch_size%100== 0 then 
        testset.data = fake; testset.labels = torch.ones(fake:size(1))
        confusion_reconstructed = test_classifier(classifier, testset)
        print(confusion_reconstructed)
      end
   end

   paths.mkdir('checkpoints')
   parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   parametersG, gradParametersG = nil, nil
   torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG:clearState())
   torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD:clearState())
   parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
   parametersG, gradParametersG = netG:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end
