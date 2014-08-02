module Octane
	class Dataset
		include Enumerable

		def initialize
			@dataset=[]
		end

		def each
			@dataset.each {|e| yield e}
		end

		def add_sample(sample,expected)
			@dataset<<[sample,expected.is_a?(Array) ? expected : [expected]]
		end

		def data
			@dataset
		end

		def batch(size)
			@dataset.sample(size)
		end

		def data=(data)
			@dataset=data
		end

		def clear
			@dataset=[]
		end

		def clone
			new_dataset=Dataset.new
			new_dataset.data=Marshal.load(Marshal.dump(@dataset.clone)) # TODO: god that's ugly.
			new_dataset
		end

		def size
			@dataset.size
		end

		def sample(sz=1)
			if sz==1
				@dataset.sample
			else
				@dataset.sample sz
			end
		end
	end

	class RegressionDataset < Dataset; end
	
	class ClassificationDataset < Dataset
		def initialize(labels)
			super()
			n=labels.size
			zeroes=[0.0]*n

			@labels=Hash[n.times.map {|lab|  a=zeroes.clone; a[lab]=1.0; [labels[lab],a]}]
		end

		def add_sample sample, expected
			@dataset<<[sample, @labels[expected]]
		end

		def split(train=0.9)
			train=(0.9*@dataset.size).to_i

			rnd=@dataset.shuffle
			train_data=rnd[0...train]
			test_data=rnd[train..rnd.size]

			train_dataset=ClassificationDataset.new(@labels.keys)
			train_dataset.data=train_data

			test_dataset=ClassificationDataset.new(@labels.keys)
			test_dataset.data=test_data

			[train_dataset, test_dataset]
		end
	end

end