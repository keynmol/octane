require 'octane'
require 'minitest/autorun'
include Octane
require './test_helper'

class TestOctane < MiniTest::Unit::TestCase

	def setup
		@single_layer=Network.new 
		@single_layer.add_layer(2) # two inputs
		@single_layer.add_layer(4) # five hidden neurons
		@single_layer.add_layer(1) # one output

		@multi_layer=Network.new 
		@multi_layer.add_layer(2) # two inputs
		@multi_layer.add_layer(4) # five hidden neurons
		@multi_layer.add_layer(4) # five hidden neurons
		@multi_layer.add_layer(1) # one output

		function = ->(x,y) { Math.cos(x+y) } # let's try to approximate cosine of sum of two variables 

		@test_dataset=Dataset.new
		@train_dataset=Dataset.new

		# create two datasets, 100 and 10000 samples respectively
		[@test_dataset, @train_dataset].zip([100, 1000]).each do |dataset, samples| 
		    samples.times do 
		        input=[rand, rand]
		        dataset.add_sample(input, function[*input]) 
		    end
		end
	end

	def test_that_network_learns
		before_training=@single_layer.test(@test_dataset)
		@single_layer.train(@train_dataset)
		after_training=@single_layer.test(@test_dataset)

		puts "Single layer: Before: #{before_training}, after: #{after_training}"
		assert_operator before_training, :> , after_training

		before_training=@multi_layer.test(@test_dataset)
		@multi_layer.train(@train_dataset)
		after_training=@multi_layer.test(@test_dataset)
		
		puts "Multi-layered: Before: #{before_training}, after: #{after_training}"
		assert_operator before_training, :> , after_training

	end

	def test_disabling_units
		@single_layer.hidden_layer.disable(0)
		assert @single_layer.hidden_layer.unit(0).disabled

		before_training=@single_layer.hidden_layer.unit(0).input_weights
		@single_layer.train(@train_dataset)
		after_training=@single_layer.hidden_layer.unit(0).input_weights

		assert_equal before_training, after_training

		@single_layer.hidden_layer.enable(0)
		refute @single_layer.hidden_layer.unit(0).disabled

		before_training=@single_layer.hidden_layer.unit(0).input_weights
		@single_layer.train(@train_dataset)
		after_training=@single_layer.hidden_layer.unit(0).input_weights

		refute_equal before_training, after_training
	end

	def test_that_dropout_works
		no_dropout_nn=Network.new 0.01
		no_dropout_nn.add_layer(2)
		no_dropout_nn.add_layer(10)
		# no_dropout_nn.add_layer(100)
		no_dropout_nn.add_layer(1, :linear)

		dropout_nn=Network.new 0.01
		dropout_nn.add_layer(2)
		dropout_nn.add_layer(10)
		# dropout_nn.add_layer(100)
		dropout_nn.add_layer(1, :linear)
		

		1000.times do |i|
			no_dropout_nn.hidden_layer.size.times do |neuron|
				if rand>0.5
					dropout_nn.hidden_layer.disable(neuron)
				end
			end

			sample=@train_dataset.sample
			
			dropout_nn.train_one(sample)
			no_dropout_nn.train_one(sample)

			dropout_nn.hidden_layer.enable_all
		end

		
		dropout_nn.output_layer.each {|unit| unit.input_weights.map! {|w| w/2}; unit.input_weights[unit.input_weights.length-1]*=2;}

		puts "Without dropout: #{no_dropout_nn.test(@test_dataset)}"
		puts "With dropout: #{dropout_nn.test(@test_dataset)}"
	end

	def test_batch_learning

		
		@net=Network.new(0.1)
		@net.add_layer(2)
		@net.add_layer(4)
		@net.add_layer(1)


		before_training=@net.test(@test_dataset)
		
		100.times do
			sample=@train_dataset.sample 10
			@net.train_batch(sample)
		end

		after_training=@net.test(@test_dataset)

		puts "Batch learning: Before: #{before_training}, after: #{after_training}"
		assert_operator before_training, :>, after_training


	end

	def dot(a1,a2)
		a1.zip(a2).map{|v1,v2| v1*v2}.reduce(:+)
	end

	def test_computation
		srand 1234

		@net=Network.new
		@net.add_layer(3)
		@net.add_layer(4)
		@net.add_layer(1,:linear)

		input=[1,2,3]
		biased=[1,2,3]+[1.0]


		@net.forward_pass([1,2,3])

		assert_equal @net.input_layer.map(&:last_squashed), input
		assert_equal @net.hidden_layer.map(&:last_activity), @net.hidden_layer.map{|neuron| dot(neuron.input_weights, biased)}
		assert_equal @net.hidden_layer.map(&:last_squashed), @net.hidden_layer.map{|neuron| Neuron::SQUASH_FUNCTIONS[neuron.type][neuron.last_activity]}

		hidden_layer_output=@net.hidden_layer.map(&:last_squashed)
		hidden_layer_activities=@net.hidden_layer.map(&:last_activity)

		output_layer_activities=@net.output_layer.map(&:last_activity)
		output_layer_output=@net.output_layer.map(&:last_squashed)

		assert_equal output_layer_activities, @net.output_layer.map{|neuron| dot(neuron.input_weights, hidden_layer_output+[1.0])}
		assert_equal output_layer_output, @net.output_layer.map{|neuron| Neuron::SQUASH_FUNCTIONS[neuron.type][neuron.last_activity]}

	end

	def test_classification
		dataset=ClassificationDataset.new %w{setosa versicolor virginica}
		Iris.dataset.each {|sample, output|
			dataset.add_sample sample, output
		}

		train_dataset, test_dataset=dataset.split(0.7)

		@net=Network.new(0.1, 0.0001)
		@net.add_layer(4)
		@net.add_layer(5)
		@net.add_layer(3, :sigmoid)

		# plot(@net)
		averages=[0.0]*4
		dataset.data.map{|sample, output| averages=averages.zip(sample).map{|a,b| a+b}}
		averages.map!{|a| a/dataset.size}

		transformation=->(input){input.zip(averages).map{|a,b| a-b}}

		@net.input_transformation=transformation

		before_training = @net.test(test_dataset)[:correct]
		@net.train(train_dataset, iterations: 100)
		after_training= @net.test(test_dataset)[:correct]

		puts "Before training: #{before_training} correct"
		puts "After training: #{after_training} correct"
		assert_operator after_training, :>=, before_training		
	end

	def test_weight_norm
		@net=Network.new(0.1, 0.00, 5)
		@net.add_layer(2)
		@net.add_layer(4)
		@net.add_layer(1,:linear)

		before_training=@net.test(@test_dataset)

		100.times do
			sample=@train_dataset.sample
			
			@net.train_one(sample)
		end

		after_training=@net.test(@test_dataset)

		assert_operator before_training, :>=, after_training

	end

	def test_autoencoder
		input_size=100
		rep_size=25

		@autoencoder=AutoEncoder.new(input_size, rep_size, false, 
									learning_rate: 0.01, 
									weight_decay: 0.00001, 
									code_layer_type: :tanh, 
									output_layer_type: :linear)

		generate_input=->{input_size.times.map {rand-0.5} }
		error=->(input, reconstruction){input.zip(reconstruction).map {|i, r| (i-r)**2}.reduce(:+)/(2*input.length)}

		@test_set=Dataset.new
		before_training=0.0

		1000.times do |i|
			before_training=@autoencoder.test(@test_set) if i==100

			input=generate_input[]
			code=@autoencoder.encode(input)
			reconstruction=@autoencoder.decode(code)

			@autoencoder.train_one(input, input) if i>100
			@test_set.add_sample input, input if i<100
		end

		after_training=@autoencoder.test(@test_set)

		assert_operator before_training, :>, after_training
	end

	def test_denoising_autoencoder
		input_size=100
		rep_size=25
		corrupt=0.5

		@autoencoder=AutoEncoder.new(input_size, rep_size, false, 
									learning_rate: 0.01, 
									weight_decay: 0.0001, 
									code_layer_type: :tanh, 
									output_layer_type: :linear)

		generate_input=->{input_size.times.map {rand-0.5} }
		error=->(input, reconstruction){input.zip(reconstruction).map {|i, r| (i-r)**2}.reduce(:+)/(2*input.length)}
		
		@test_set=Dataset.new
		before_training=0.0

		1000.times do |i|
			before_training=@autoencoder.test(@test_set) if i==100
			input=generate_input[]

			mask=Array.new((corrupt*input_size).to_i, 0)
			mask+=Array.new(input_size-mask.length, 1)

			mask.shuffle!

			masked_input=input.zip(mask).map{|i,m| i*m}

			@autoencoder.train_one(masked_input, input) if i>100
			@test_set.add_sample masked_input, input if i<100
		end

		after_training=@autoencoder.test(@test_set)

		assert_operator before_training, :>, after_training
	end
end